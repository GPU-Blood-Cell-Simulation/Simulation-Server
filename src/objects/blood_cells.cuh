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
		{bloodCellCount, 2}//,
		//{bloodCellCount, 3}
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
	void gatherForcesFromNeighbors(int gpuId, const std::array<cudaStream_t, bloodCellTypeCount>& streams);

	void propagateForcesIntoPositions(int blocks, int threadsPerBlock);

	#ifdef MULTI_GPU
	void broadcastParticles(ncclComm_t* comms, cudaStream_t* streams);
	void reduceForces(ncclComm_t* comms, cudaStream_t* streams);
	#endif
};

#endif