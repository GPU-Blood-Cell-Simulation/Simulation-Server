#include "blood_cells.cuh"

#include "../simulation/physics.cuh"
#include "../meta_factory/blood_cell_factory.hpp"
#include "../utilities/cuda_handle_error.cuh"
#include "../utilities/math.cuh"
#include "../utilities/cuda_threads.hpp"

#include <vector>

#include "cuda_runtime.h"

constexpr float NO_SPRING = 0;

BloodCells::BloodCells()
{
	HANDLE_ERROR(cudaMalloc(&dev_springGraph, sizeof(float) * totalGraphSize));
	HANDLE_ERROR(cudaMemcpy(dev_springGraph, springGraph.data(), sizeof(float) * totalGraphSize, cudaMemcpyHostToDevice));
}

BloodCells::BloodCells(const BloodCells& other) : isCopy(true), particles(other.particles), dev_springGraph(other.dev_springGraph) {}

BloodCells::~BloodCells()
{
	if (!isCopy)
	{
		HANDLE_ERROR(cudaFree(dev_springGraph));
	}
}

/// <summary>
/// Adjust the force acting on every particle based on the forces applied to its neighbors connected by springs
/// </summary>
template<int bloodCellCount, int particlesInBloodCell, int particlesStart, int springGraphStart>
__global__ static void gatherForcesKernel(BloodCells bloodCells)
{
	int indexInType = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexInType >= particlesInBloodCell * bloodCellCount)
		return;

	int indexInCell = indexInType % particlesInBloodCell;
	int realIndex = particlesStart + indexInType;

	float3 position = bloodCells.particles.positions.get(realIndex);
	float3 velocity = bloodCells.particles.velocities.get(realIndex);
	float3 initialForce = bloodCells.particles.forces.get(realIndex);
	float3 newPosition {0,0,0}, newVelocity {0,0,0}, newForce{ 0, 0, 0 };

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
	newForce = newForce + physics::accumulateEnvironmentForcesForParticles(velocity);

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
	mp_for_each<IndexList>([&](auto i)
		{
			using BloodCellDefinition = mp_at_c<BloodCellList, i>;
			constexpr int particlesStart = particleStarts[i];
			constexpr int graphStart = accumulatedGraphSizes[i];

			CudaThreads threads(BloodCellDefinition::count * BloodCellDefinition::particlesInCell);
			gatherForcesKernel<BloodCellDefinition::count, BloodCellDefinition::particlesInCell, particlesStart, graphStart>
				<< <threads.blocks, threads.threadsPerBlock, 0, streams[i] >> > (*this);
		});
}