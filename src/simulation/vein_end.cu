#include "vein_end.cuh"

#include "../config/physics.hpp"
#include "../config/simulation.hpp"
#include "../meta_factory/blood_cell_factory.hpp"
#include "../utilities/math.cuh"

#include "cuda_runtime.h"


constexpr float upperBoundTreshold = 0.95f * height;
constexpr float lowerBoundTreshold = 0.2f * height;
constexpr float targetTeleportHeight = 0.85f * height;

enum SynchronizationType { warpSync, blockSync };


constexpr SynchronizationType SelectSynchronizationType(int bloodCellsCnt, int particlesInBloodCell)
{
	if (bloodCellsCnt * particlesInBloodCell <= CudaThreads::threadsInWarp ||
		CudaThreads::threadsInWarp % particlesInBloodCell == 0)
		return warpSync;

	return blockSync;
}

constexpr int CalculateThreadsPerBlock(SynchronizationType syncType, int bloodCellsCnt, int particlesInBloodCell)
{
	switch (syncType)
	{
	case warpSync:
		if (bloodCellsCnt * particlesInBloodCell < CudaThreads::threadsInWarp)
			return bloodCellsCnt * particlesInBloodCell;

		// Max number of full warps
		return (CudaThreads::maxThreadsInBlock / CudaThreads::threadsInWarp) * CudaThreads::threadsInWarp;

	case blockSync:
		// Max mulitple of number of particles in blood cell
		return (CudaThreads::maxThreadsInBlock / particlesInBloodCell) * particlesInBloodCell;

	default:
		throw std::domain_error("Unknown synchronization type");
	}
}

constexpr int CalculateBlocksCount(SynchronizationType syncType, int particleCount, int particlesInBloodCell)
{
	return constCeil(static_cast<float>(particleCount) / CalculateThreadsPerBlock(syncType, particleCount, particlesInBloodCell));
}


template <int bloodCellsCount, int particlesInBloodCell, int particlesStart>
__global__ void handleVeinEndsBlockSync(BloodCells bloodCells)
{
	__shared__ bool belowVein[CalculateThreadsPerBlock(blockSync, bloodCellsCount, particlesInBloodCell)];
	int indexInType = blockDim.x * blockIdx.x + threadIdx.x;
	 
	if (indexInType >= bloodCellsCount * particlesInBloodCell)
		return;

	int realIndex = particlesStart + indexInType;
	float posY = bloodCells.particles.positions.y[realIndex]; 

	if (posY >= upperBoundTreshold) {
		// Bounce particle off upper bound
		bloodCells.particles.velocities.y[realIndex] -= 5;
	}

	// Check lower bound
	bool teleport = (posY <= lowerBoundTreshold);
	belowVein[threadIdx.x] = teleport;

	__syncthreads();

	int particleInCellIndex = realIndex % particlesInBloodCell;
	int numberOfParticlesInThread = threadIdx.x / particlesInBloodCell * particlesInBloodCell;

	// Algorithm goes through all neighbours and checks if any of them is low enought to be teleported
	#pragma unroll
	for (int i = 1; i < particlesInBloodCell; i++)
	{
 		teleport |= belowVein[((particleInCellIndex + i) % particlesInBloodCell) + numberOfParticlesInThread];
	}

	if (teleport)
	{
		// TODO: add some randomnes to velocity and change positivon to one which is always inside the vein
		bloodCells.particles.positions.y[realIndex] = targetTeleportHeight;
		bloodCells.particles.velocities.set(realIndex, make_float3(initVelocityX, initVelocityY, initVelocityZ));
	}
}

# include <stdio.h>
template <int bloodCellsCount, int particlesInBloodCell, int particlesStart>
__global__ void handleVeinEndsWarpSync(BloodCells bloodCells)
{
  	int indexInType = blockDim.x * blockIdx.x + threadIdx.x;

	//printf("%d\n", indexInType);
	if (indexInType >= bloodCellsCount * particlesInBloodCell)
		return;

	int threadInWarpID = threadIdx.x % CudaThreads::threadsInWarp;

	int realIndex = particlesStart + indexInType;
	float posY = bloodCells.particles.positions.y[realIndex];

	if (posY >= upperBoundTreshold) {
		// Bounce particle off upper bound
		bloodCells.particles.velocities.y[realIndex] -= 5;
	}

	static constexpr int initSyncBitMask = (particlesInBloodCell == 32) ? 0xffffffff : (1 << (particlesInBloodCell)) - 1;
	int syncBitMask = initSyncBitMask << static_cast<int>(std::floor(static_cast<float>(threadInWarpID)/particlesInBloodCell)) * particlesInBloodCell;

	// Bit mask of particles, which are below treshold
	int particlesBelowTreshold = __any_sync(syncBitMask, posY <= lowerBoundTreshold);

	if (particlesBelowTreshold != 0) {
		// TODO: add some randomnes to velocity and change positivon to one which is always inside the vein
		bloodCells.particles.positions.y[realIndex] = targetTeleportHeight;
		bloodCells.particles.velocities.set(realIndex, make_float3(initVelocityX, initVelocityY, initVelocityZ));
	}
}


void HandleVeinEnd(BloodCells& cells, const std::array<cudaStream_t, bloodCellTypeCount>& streams)
{ 
	using IndexList = mp_iota_c<bloodCellTypeCount>;
	mp_for_each<IndexList>([&](auto i)
		{
			using BloodCellDefinition = mp_at_c<BloodCellList, i>;
			constexpr int particlesStart = particlesStarts[i];

			constexpr SynchronizationType syncType = SelectSynchronizationType(
				BloodCellDefinition::count,
				BloodCellDefinition::particlesInCell
			);

			constexpr int threadsPerBlock = CalculateThreadsPerBlock(
				syncType,
				BloodCellDefinition::count,
				BloodCellDefinition::particlesInCell
			);

			constexpr int blocksCnt = CalculateBlocksCount(
				syncType,
				BloodCellDefinition::count * BloodCellDefinition::particlesInCell,
				BloodCellDefinition::particlesInCell
			);

			if constexpr (syncType == warpSync)
				handleVeinEndsWarpSync<BloodCellDefinition::count, BloodCellDefinition::particlesInCell, particlesStart>
				<< <blocksCnt, threadsPerBlock, 0, streams[i] >> > (cells);
			else if constexpr (syncType == blockSync)
				handleVeinEndsBlockSync<BloodCellDefinition::count, BloodCellDefinition::particlesInCell, particlesStart>
				<< <blocksCnt, threadsPerBlock, 0, streams[i] >> > (cells);
			else
				static_assert(false, "Unknown synchronization type");
		});
}
