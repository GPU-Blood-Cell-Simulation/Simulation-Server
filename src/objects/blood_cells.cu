#include "blood_cells.cuh"

#include "../simulation/physics.cuh"
#include "../meta_factory/blood_cell_factory.hpp"
#include "../utilities/cuda_handle_error.cuh"
#include "../utilities/math.cuh"
#include "../utilities/cuda_threads.hpp"

#include <vector>
#include <iostream>

#include "cuda_runtime.h"

constexpr float NO_SPRING = 0;

BloodCells::BloodCells()
{
	for (int i = 0; i < gpuCount; i++)
	{
		cudaSetDevice(i);
		CUDACHECK(cudaMalloc(&(dev_springGraph[i]), sizeof(float) * totalGraphSize));
		CUDACHECK(cudaMemcpy(dev_springGraph[i], springGraph.data(), sizeof(float) * totalGraphSize, cudaMemcpyHostToDevice));
		CUDACHECK(cudaMalloc(&(initialRadiuses[i]), sizeof(float) * particleDistinctCellsCount));
	}
	cudaSetDevice(0);
}

BloodCells::BloodCells(const BloodCells& other) : isCopy(true), particles(other.particles), dev_springGraph(other.dev_springGraph), particleCenters(other.particleCenters), initialRadiuses(other.initialRadiuses) {}

BloodCells::~BloodCells()
{
	if (!isCopy)
	{
		for (int i = 0; i < gpuCount; i++)
		{
			CUDACHECK(cudaSetDevice(i));
			CUDACHECK(cudaFree(initialRadiuses[i]));
			CUDACHECK(cudaFree(dev_springGraph[i]));
		}
		CUDACHECK(cudaSetDevice(0));
	}
}

template<int bloodCellCount, int particlesInBloodCell, int particlesStart, int bloodCellStart>
__global__ static void calculateBloodCellsCenters(int gpuId, int gpuStart, int gpuEnd, BloodCells bloodCells)
{
	int relativeCellIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if(relativeCellIndex >= bloodCellCount)
		return;	
	int realCellIndex = bloodCellStart + relativeCellIndex;
	int firstParticleIndex = particlesStart + relativeCellIndex * particlesInBloodCell;
	float3 center = make_float3(0,0,0);
	for(int id = 0; id < particlesInBloodCell; ++id)
	{
		int particleId = firstParticleIndex + id;
		center = center + bloodCells.particles.positions[gpuId].get(particleId);
	}
	bloodCells.particleCenters[gpuId].set(realCellIndex, center/particlesInBloodCell);
}

/// <summary>
/// Adjust the force acting on every particle based on the forces applied to its neighbors connected by springs
/// </summary>
template<int bloodCellCount, int particlesInBloodCell, int particlesStart, int bloodCellStart, int bloodCellModelStart, int springGraphStart>
__global__ static void gatherForcesKernel(int gpuId, int gpuStart, int gpuEnd, BloodCells bloodCells)
{
	int indexInType = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexInType >= particlesInBloodCell * bloodCellCount || indexInType < gpuStart || indexInType >= gpuEnd)
		return;

	int indexInCell = indexInType % particlesInBloodCell;
	int realIndex = particlesStart + indexInType;
	int cellRealIndex = bloodCellStart + indexInType / particlesInBloodCell;

	float3 position = bloodCells.particles.positions[gpuId].get(realIndex);
	float3 velocity = bloodCells.particles.velocities[gpuId].get(realIndex);
	float3 initialForce = bloodCells.particles.forces[gpuId].get(realIndex);
	float3 newForce{ 0, 0, 0 };
	float3 radius = position - bloodCells.particleCenters[gpuId].get(cellRealIndex);
	float initialRadius = bloodCells.initialRadiuses[gpuId][bloodCellModelStart + indexInCell];

#ifdef USE_RUNGE_KUTTA_FOR_PARTICLE
	float3 newPosition {0,0,0}, newVelocity {0,0,0};
#endif

#pragma unroll
	for (int neighbourCellindex = 0; neighbourCellindex < particlesInBloodCell; neighbourCellindex++)
	{
		float springLength = bloodCells.dev_springGraph[gpuId][springGraphStart + neighbourCellindex * particlesInBloodCell + indexInCell];

		if (springLength != NO_SPRING)
		{
			int neighbourIndex = realIndex - indexInCell + neighbourCellindex;

			float3 neighbourPosition = bloodCells.particles.positions[gpuId].get(neighbourIndex);
			float3 neighbourVelocity = bloodCells.particles.velocities[gpuId].get(neighbourIndex);
			float3 neighbourInitialForce = bloodCells.particles.forces[gpuId].get(neighbourIndex);
			float3 p{ 0,0,0 }, v{ 0,0,0 };
			float3 springForceComponent = physics::calculateParticlesSpringForceComponent(position - neighbourPosition,
				velocity - neighbourVelocity, initialForce, neighbourInitialForce, springLength, p, v);

#ifdef USE_RUNGE_KUTTA_FOR_PARTICLE
			newPosition = newPosition + p;
			newVelocity = newVelocity + v;
#endif
			newForce = newForce + springForceComponent;
		}
	}

	// add gravitation and viscous damping
	newForce = newForce + physics::accumulateEnvironmentForcesForParticles(velocity, length(radius)/initialRadius);

#ifdef USE_RUNGE_KUTTA_FOR_PARTICLE
	//bloodCells.particles.positions.add(realIndex, newPosition);
	//bloodCells.particles.velocities.add(realIndex, newVelocity);
	bloodCells.particles.forces[gpuId].set(realIndex, (initialForce + newForce)/6.0f);
#else
	bloodCells.particles.forces[gpuId].set(realIndex, (initialForce + newForce) / 2);
#endif
}

void BloodCells::gatherForcesFromNeighbors(int gpuId, int bloodCellGpuStart, int bloodCellGpuEnd,
	int particleGpuStart, int particleGpuEnd, const std::array<cudaStream_t, bloodCellTypeCount>& streams)
{
	using IndexList = mp_iota_c<bloodCellTypeCount>;
	
	// recalculate centers
	mp_for_each<IndexList>([&](auto i)
	{
		using BloodCellDefinition = mp_at_c<BloodCellList,i>;
		static constexpr int particlesStart = particleStarts[i];
		static constexpr int bloodCellStart = bloodCellTypesStarts[i];

		static CudaThreads threads(BloodCellDefinition::count);
		calculateBloodCellsCenters<BloodCellDefinition::count, BloodCellDefinition::particlesInCell, particlesStart, bloodCellStart>
			<<<threads.blocks, threads.threadsPerBlock, 0, streams[i]>>>(gpuId, bloodCellGpuStart, bloodCellGpuEnd, *this);
	});
	CUDACHECK(cudaDeviceSynchronize());
	
	// gather forces
	mp_for_each<IndexList>([&](auto i)
		{
			using BloodCellDefinition = mp_at_c<BloodCellList, i>;
			static constexpr int particlesStart = particleStarts[i];
			static constexpr int bloodCellStart = bloodCellTypesStarts[i];
			static constexpr int bloodCellModelStart = bloodCellModelStarts[i];
			static constexpr int graphStart = accumulatedGraphSizes[i];

			CudaThreads threads(BloodCellDefinition::count * BloodCellDefinition::particlesInCell);
			gatherForcesKernel<BloodCellDefinition::count, BloodCellDefinition::particlesInCell, particlesStart, bloodCellStart, bloodCellModelStart, graphStart>
				<< <threads.blocks, threads.threadsPerBlock, 0, streams[i] >> > (gpuId, particleGpuStart, particleGpuEnd, *this);
		});
}

__global__ void propagateParticleForcesKernel(Particles particles)
{
	int particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if(particleId >= particleCount)
		return;

	float3 F = particles.forces[0].get(particleId);
	float3 initialVelocity = particles.velocities[0].get(particleId);
	// propagate particle forces into velocities
	float3 velocity = initialVelocity + dt * F;
	particles.velocities[0].set(particleId, velocity);

		// propagate velocities into positions
	#ifdef USE_RUNGE_KUTTA_FOR_PARTICLE
		float3 k1_x = dt * velocity;
		float3 k2_x = dt * (velocity + k1_x / 2);
		float3 k3_x = dt * (velocity + k2_x / 2);
		float3 k4_x = dt * (velocity + k3_x);
		particles.positions[0].add(particleId, (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6.0f);
	#else
		// using Heun's method
		particles.positions[0].add(particleId, 0.5f * dt * (velocity + initialVelocity));
	#endif
}

void BloodCells::propagateForcesIntoPositions(int blocks, int threadsPerBlock)
{
	propagateParticleForcesKernel<<<blocks, threadsPerBlock>>>(particles);
}

#ifdef MULTI_GPU

using namespace nccl;

void BloodCells::broadcastParticles(ncclComm_t* comms, cudaStream_t* streams)
{
	// Broadcast positions
	NCCLCHECK(ncclGroupStart());
	broadcast(particles.positions, particleCount, ncclFloat, comms, streams);
	broadcast(particles.velocities, particleCount, ncclFloat, comms, streams);
	NCCLCHECK(ncclGroupEnd());

	// Manually zeroing forces on each gpu is cheaper than broadcasting them
	for (int i = 0; i < gpuCount; i++)
	{
		CUDACHECK(cudaSetDevice(i));
		CUDACHECK(cudaMemset(particles.forces[i].x, 0, particleCount * sizeof(float)));
		CUDACHECK(cudaMemset(particles.forces[i].y, 0, particleCount * sizeof(float)));
		CUDACHECK(cudaMemset(particles.forces[i].z, 0, particleCount * sizeof(float)));

		CUDACHECK(cudaStreamSynchronize(streams[i]));
	}
	CUDACHECK(cudaSetDevice(0));
}

__global__ void debugKernel(int gpuId, Particles particles)
{
	int particleId = particleCount - 1 - blockIdx.x * blockDim.x + threadIdx.x;

	printf("force gpu:%d, %f ", gpuId, particles.forces[gpuId].get(particleId).x);
}

void BloodCells::reduceForces(ncclComm_t* comms, cudaStream_t* streams)
{
	cudaSetDevice(2);
	debugKernel<<<1, 400>>>(2, particles);
	cudaDeviceSynchronize();
	std::cout << "\n---------------------------\n";
	
	NCCLCHECK(ncclGroupStart());
	reduce(particles.forces, particleCount, ncclFloat, comms, streams);
	NCCLCHECK(ncclGroupEnd());
	sync(streams);
	cudaDeviceSynchronize();

	
	// cudaSetDevice(0);
	// debugKernel<<<1, 20>>>(0, particles);
	// cudaDeviceSynchronize();

}
#endif