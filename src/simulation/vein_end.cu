#include "vein_end.cuh"

#include "../simulation/physics.cuh"
#include "../config/simulation.hpp"
#include "../meta_factory/blood_cell_factory.hpp"
#include "../meta_factory/vein_factory.hpp"
#include "../utilities/math.cuh"

#include "cuda_runtime.h"


constexpr float upperBoundTreshold = maxY - gridYMargin / 2;
constexpr float lowerBoundTreshold = minY + gridYMargin / 2;
constexpr float rightBoundTreshold = maxX - gridXZMargin / 2;
constexpr float leftBoundTreshold = minX + gridXZMargin / 2;
constexpr float frontBoundTreshold = maxZ - gridXZMargin / 2;
constexpr float backBoundTreshold = minZ + gridXZMargin / 2;
constexpr float targetTeleportHeight = minSpawnY;

enum SynchronizationType { warpSync, blockSync };


constexpr SynchronizationType SelectSynchronizationType(int bloodCellsCnt, int particlesInBloodCell)
{
	if (bloodCellsCnt * particlesInBloodCell <= CudaThreads::threadsInWarp ||
		CudaThreads::threadsInWarp % particlesInBloodCell == 0)
		return warpSync;

	return blockSync;
}

constexpr __host__ __device__ int CalculateThreadsPerBlock(SynchronizationType syncType, int bloodCellsCnt, int particlesInBloodCell)
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
		return -1;
	}
}

constexpr int CalculateBlocksCount(SynchronizationType syncType, int particleCount, int particlesInBloodCell)
{
	return constCeil(static_cast<float>(particleCount) / CalculateThreadsPerBlock(syncType, particleCount, particlesInBloodCell));
}

template <int bloodCellsCount, int particlesInBloodCell, int particlesStart, int bloodCellmodelStart>
__global__ void handleVeinEndsBlockSync(BloodCells bloodCells, curandState* states, cudaVec3 bloodCellModels)
{ 
	__shared__ bool belowVein[CalculateThreadsPerBlock(blockSync, bloodCellsCount, particlesInBloodCell)];

	int indexInType = blockDim.x * blockIdx.x + threadIdx.x;

	if (indexInType >= bloodCellsCount * particlesInBloodCell)
		return;

	int realIndex = particlesStart + indexInType;
	float posX = bloodCells.particles.positions[0].x[realIndex];
	float posY = bloodCells.particles.positions[0].y[realIndex]; 
	float posZ = bloodCells.particles.positions[0].z[realIndex];

	if (posY >= upperBoundTreshold) {
		// Bounce particle off upper bound
		bloodCells.particles.velocities[0].y[realIndex] -= 5;
	}

	// Check if teleportiation should occur
	bool teleport = false;
	mp_for_each<mp_iota_c<veinEndingCenterCount>>([&](auto i)
		{
			static constexpr float endingX = mp_at_c<VeinEndingCenters, i>::x;
			static constexpr float endingY = mp_at_c<VeinEndingCenters, i>::y;
			static constexpr float endingZ = mp_at_c<VeinEndingCenters, i>::z;
			static constexpr float radius = mp_at_c<VeinEndingRadii, i>::value;
			teleport = teleport || length_squared(float3{posX - endingX, posY - endingY, posZ - endingZ}) <= radius * radius;
		});
	

	// Check additional constraints to make sure particles don't leave the grid
	teleport = teleport || posY <= lowerBoundTreshold || posX <= leftBoundTreshold || posX >= rightBoundTreshold || posZ <= backBoundTreshold || posZ >= frontBoundTreshold;
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
		bloodCells.particles.positions[0].x[realIndex] = (curand_uniform(&states[realIndex/particlesInBloodCell]) - 0.5f) * 1.2f * cylinderRadius + bloodCellModels.x[bloodCellmodelStart + indexInType % particlesInBloodCell] - bloodCellModels.x[bloodCellmodelStart];
		bloodCells.particles.positions[0].y[realIndex] = targetTeleportHeight + bloodCellModels.y[bloodCellmodelStart + indexInType % particlesInBloodCell] - bloodCellModels.y[bloodCellmodelStart];
		bloodCells.particles.positions[0].z[realIndex] = (curand_uniform(&states[realIndex/particlesInBloodCell]) - 0.5f) * 1.2f * cylinderRadius + bloodCellModels.z[bloodCellmodelStart + indexInType % particlesInBloodCell] - bloodCellModels.z[bloodCellmodelStart];
		bloodCells.particles.velocities[0].set(realIndex, make_float3(initVelocityX, initVelocityY, initVelocityZ));
	}
}


template <int bloodCellsCount, int particlesInBloodCell, int particlesStart, int bloodCellmodelStart>
__global__ void handleVeinEndsWarpSync(BloodCells bloodCells, curandState* states, cudaVec3 bloodCellModels)
{
	int indexInType = blockDim.x * blockIdx.x + threadIdx.x;

	if (indexInType >= bloodCellsCount * particlesInBloodCell)
		return;

	int threadInWarpID = threadIdx.x % CudaThreads::threadsInWarp;

	int realIndex = particlesStart + indexInType;
	float posX = bloodCells.particles.positions[0].x[realIndex];
	float posY = bloodCells.particles.positions[0].y[realIndex]; 
	float posZ = bloodCells.particles.positions[0].z[realIndex]; 

	if (posY >= upperBoundTreshold) {
		// Bounce particle off upper bound
		bloodCells.particles.velocities[0].y[realIndex] -= 5;
	}

	static constexpr int initSyncBitMask = (particlesInBloodCell == 32) ? 0xffffffff : (1 << (particlesInBloodCell)) - 1;
	int syncBitMask = initSyncBitMask << static_cast<int>(std::floor(static_cast<float>(threadInWarpID) / particlesInBloodCell)) * particlesInBloodCell;

	// Bit mask of particles, which are below treshold
	int particlesBelowTreshold = __any_sync(syncBitMask, posY <= lowerBoundTreshold || posX <= leftBoundTreshold || posX >= rightBoundTreshold || posZ <= backBoundTreshold || posZ >= frontBoundTreshold);

	if (particlesBelowTreshold != 0) {
		bloodCells.particles.positions[0].x[realIndex] = (curand_uniform(&states[realIndex/particlesInBloodCell]) - 0.5f) * 1.2f * cylinderRadius + bloodCellModels.x[bloodCellmodelStart + indexInType % particlesInBloodCell] - bloodCellModels.x[bloodCellmodelStart];
		bloodCells.particles.positions[0].y[realIndex] = targetTeleportHeight + bloodCellModels.y[bloodCellmodelStart + indexInType % particlesInBloodCell] - bloodCellModels.y[bloodCellmodelStart];
		bloodCells.particles.positions[0].z[realIndex] = (curand_uniform(&states[realIndex/particlesInBloodCell]) - 0.5f) * 1.2f * cylinderRadius + bloodCellModels.z[bloodCellmodelStart + indexInType % particlesInBloodCell] - bloodCellModels.z[bloodCellmodelStart];
		bloodCells.particles.velocities[0].set(realIndex, make_float3(initVelocityX, initVelocityY, initVelocityZ));
	}
}


void HandleVeinEnd(BloodCells& cells, curandState* devStates, const std::array<cudaStream_t, bloodCellTypeCount>& streams, cudaVec3& bloodCellModels)
{
	using IndexList = mp_iota_c<bloodCellTypeCount>;
	mp_for_each<IndexList>([&](auto i)
		{
			using BloodCellDefinition = mp_at_c<BloodCellList, i>;
			constexpr int particlesStart = particleStarts[i];

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
				handleVeinEndsWarpSync<BloodCellDefinition::count, BloodCellDefinition::particlesInCell, particlesStart, bloodCellModelStarts[i]>
				<< <blocksCnt, threadsPerBlock, 0, streams[i] >> > (cells, devStates, bloodCellModels);
			else if constexpr (syncType == blockSync)
				handleVeinEndsBlockSync<BloodCellDefinition::count, BloodCellDefinition::particlesInCell, particlesStart, bloodCellModelStarts[i]>
				<< <blocksCnt, threadsPerBlock, 0, streams[i] >> > (cells, devStates, bloodCellModels);
		});
}
