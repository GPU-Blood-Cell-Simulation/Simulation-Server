#ifndef BLOOD_CELLS_H
#define BLOOD_CELLS_H

#include "../meta_factory/blood_cell_factory.hpp"
#include "particles.cuh"
#include "../utilities/host_device_array.cuh"
#include "../utilities/math.cuh"

#ifdef MULTI_GPU
#include "../utilities/nccl_operations.cuh"
#endif

/// <summary>
/// Contains all the particles and the springs graph
/// </summary>
class BloodCells
{
	bool isCopy = false;

public:
	Particles particles;
	#ifdef MULTI_GPU
	HostDeviceArray<cudaVec3, gpuCount> particleCenters { // GPU_COUNT_DEPENDENT
		{bloodCellCount, 0},
		{bloodCellCount, 1},
		{bloodCellCount, 2},
		{bloodCellCount, 3}
	};
	#else
	HostDeviceArray<cudaVec3, gpuCount> particleCenters {{bloodCellCount, 0}};
	#endif

	HostDeviceArray<float*, gpuCount> initialRadiuses;
	HostDeviceArray<float*, gpuCount> dev_springGraph;

	BloodCells();
	BloodCells(const BloodCells& other);
	~BloodCells();

	/// <summary>
	/// Perform gathering forces contribution from neighbour particles for every particle
	/// </summary>
	/// <param name="gpuId">The gpu calculating the kernels</param>
	/// <param name="bloodCellGpuStart">Start of the blood cell array range for this gpu</param>
	/// <param name="bloodCellGpuEnd">End of the blood cell array range for this gpu</param>
	/// <param name="particleGpuStart">Start of the particle array range for this gpu</param>
	/// <param name="particleGpuEnd">End of the particle array range for this gpu</param>
	/// <param name="streams">CUDA streams to be used for different blood cell types</param>
	void gatherForcesFromNeighbors(int gpuId, int bloodCellGpuStart, int bloodCellGpuEnd,
		int particleGpuStart, int particleGpuEnd, const std::array<cudaStream_t, bloodCellTypeCount>& streams);

	/// <summary>
	/// Transform forces into velocities and velocities into positions
	/// </summary>
	/// <param name="blocks">CUDA block count</param>
	/// <param name="blocks">CUDA threads per one block</param>
	void propagateForcesIntoPositions(int blocks, int threadsPerBlock);

	#ifdef MULTI_GPU
	/// <summary>
	/// Broadcast particle data from the root gpu to all others
	/// </summary>
	/// <param name="blocks">NCCL comm array</param>
	/// <param name="blocks">NCCL synchronization stream array</param>
	void broadcastParticles(ncclComm_t* comms, cudaStream_t* streams);

	/// <summary>
	/// Reduce particle forces from all gpus to the root node
	/// </summary>
	/// <param name="blocks">NCCL comm array</param>
	/// <param name="blocks">NCCL synchronization stream array</param>
	void reduceForces(ncclComm_t* comms, cudaStream_t* streams);
	#endif
};

#endif