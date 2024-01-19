#include "blood_cells.cuh"

#include "../simulation/physics.cuh"
#include "../meta_factory/blood_cell_factory.hpp"
#include "../utilities/cuda_handle_error.cuh"
#include "../utilities/math.cuh"
#include "../utilities/cuda_threads.hpp"

#include <vector>

#include "cuda_runtime.h"

constexpr float NO_SPRING = 0;

BloodCells::BloodCells(): particleCenters(bloodCellCount)
{
	CUDACHECK(cudaMalloc(&dev_springGraph, sizeof(float) * totalGraphSize));
	CUDACHECK(cudaMemcpy(dev_springGraph, springGraph.data(), sizeof(float) * totalGraphSize, cudaMemcpyHostToDevice));
	CUDACHECK(cudaMalloc(&initialRadiuses, sizeof(float)*particleDistinctCellsCount));
}

BloodCells::BloodCells(const BloodCells& other) : isCopy(true), particles(other.particles), dev_springGraph(other.dev_springGraph), particleCenters(other.particleCenters), initialRadiuses(other.initialRadiuses) {}

BloodCells::~BloodCells()
{
	if (!isCopy)
	{
		CUDACHECK(cudaFree(dev_springGraph));
	}
}

template<int bloodCellCount,  int particlesInBloodCell, int particlesStart, int bloodCellStart>
__global__ static void calculateBloodCellsCenters(BloodCells bloodCells)
{
	int relativeCellIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if(relativeCellIndex >= bloodCellCount)
		return;
	int realCellIndex = bloodCellStart + relativeCellIndex;
	int firstParticleIndex = particlesStart + relativeCellIndex*particlesInBloodCell;
	float3 center = make_float3(0,0,0);
	for(int id = 0; id < particlesInBloodCell; ++id)
	{
		int particleId = firstParticleIndex + id;
		center = center + bloodCells.particles.positions.get(particleId);
	}
	bloodCells.particleCenters.set(realCellIndex, center/particlesInBloodCell);
}

/// <summary>
/// Adjust the force acting on every particle based on the forces applied to its neighbors connected by springs
/// </summary>
template<int bloodCellCount, int particlesInBloodCell, int particlesStart, int bloodCellStart, int bloodCellModelStart, int springGraphStart>
__global__ static void gatherForcesKernel(BloodCells bloodCells)
{
	int indexInType = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexInType >= particlesInBloodCell * bloodCellCount)
		return;

	int indexInCell = indexInType % particlesInBloodCell;
	int realIndex = particlesStart + indexInType;
	int cellRealIndex = bloodCellStart + indexInType / particlesInBloodCell;

	float3 position = bloodCells.particles.positions.get(realIndex);
	float3 velocity = bloodCells.particles.velocities.get(realIndex);
	float3 initialForce = bloodCells.particles.forces.get(realIndex);
	float3 newForce{ 0, 0, 0 };
	float3 radius = position - bloodCells.particleCenters.get(cellRealIndex);
	float initialRadius =  bloodCells.initialRadiuses[bloodCellModelStart + indexInCell];

#ifdef USE_RUNGE_KUTTA_FOR_PARTICLE
	float3 newPosition {0,0,0}, newVelocity {0,0,0};
#endif

#pragma unroll
	for (int neighbourCellindex = 0; neighbourCellindex < particlesInBloodCell; neighbourCellindex++)
	{
		float springLength = bloodCells.dev_springGraph[springGraphStart + neighbourCellindex * particlesInBloodCell + indexInCell];

		if (springLength != NO_SPRING)
		{
			int neighbourIndex = realIndex - indexInCell + neighbourCellindex;

			float3 neighbourPosition = bloodCells.particles.positions.get(neighbourIndex);
			float3 neighbourVelocity = bloodCells.particles.velocities.get(neighbourIndex);
			float3 neighbourInitialForce = bloodCells.particles.forces.get(neighbourIndex);
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
	bloodCells.particles.forces.set(realIndex, (initialForce + newForce)/6.0f);
#else
	bloodCells.particles.forces.set(realIndex, (initialForce + newForce) / 2);
#endif
}

void BloodCells::gatherForcesFromNeighbors(const std::array<cudaStream_t, bloodCellTypeCount>& streams)
{
	using IndexList = mp_iota_c<bloodCellTypeCount>;
	
	// recalculate centers
	mp_for_each<IndexList>([&](auto i)
	{
		using BloodCellDefinition = mp_at_c<BloodCellList,i>;
		constexpr int particlesStart = particleStarts[i];
		constexpr int bloodCellStart = bloodCellTypesStarts[i];

		CudaThreads threads(BloodCellDefinition::count);
		calculateBloodCellsCenters<BloodCellDefinition::count, BloodCellDefinition::particlesInCell, particlesStart, bloodCellStart>
			<<<threads.blocks, threads.threadsPerBlock, 0, streams[i]>>>(*this);
	});
	CUDACHECK(cudaDeviceSynchronize());
	
	// gather forces
	mp_for_each<IndexList>([&](auto i)
		{
			using BloodCellDefinition = mp_at_c<BloodCellList, i>;
			constexpr int particlesStart = particleStarts[i];
			constexpr int bloodCellStart = bloodCellTypesStarts[i];
			constexpr int bloodCellModelStart = bloodCellModelStarts[i];
			constexpr int graphStart = accumulatedGraphSizes[i];

			CudaThreads threads(BloodCellDefinition::count * BloodCellDefinition::particlesInCell);
			gatherForcesKernel<BloodCellDefinition::count, BloodCellDefinition::particlesInCell, particlesStart, bloodCellStart, bloodCellModelStart, graphStart>
				<< <threads.blocks, threads.threadsPerBlock, 0, streams[i] >> > (*this);
		});
}

