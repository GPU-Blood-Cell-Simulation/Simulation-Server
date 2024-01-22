#pragma once

#include "../simulation/physics.cuh"
#include "../meta_factory/vein_factory.hpp"
#include "../utilities/cuda_vec3.cuh"
#include "../utilities/host_device_array.cuh"
#include "../utilities/math.cuh"
#include "../utilities/vertex_index_enum.hpp"
#include "vein_neighbors.cuh"

#include <vector>
#include <tuple>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef MULTI_GPU
#include "../utilities/nccl_operations.cuh"
#endif

/// <summary>
/// Vein triangles
/// </summary>
class VeinTriangles
{
public:
	static constexpr int vertexCount = veinPositionCount;

	#ifdef MULTI_GPU
	HostDeviceArray<cudaVec3, gpuCount> positions { // GPU_COUNT_DEPENDENT
		{vertexCount, 0},
		{vertexCount, 1},
		{vertexCount, 2}//,
		//{vertexCount, 3}
	};
	HostDeviceArray<cudaVec3, gpuCount> velocities { // GPU_COUNT_DEPENDENT
		{vertexCount, 0},
		{vertexCount, 1},
		{vertexCount, 2}//,
		//{vertexCount, 3}
	};
	HostDeviceArray<cudaVec3, gpuCount> forces{ // GPU_COUNT_DEPENDENT
		{vertexCount, 0},
		{vertexCount, 1},
		{vertexCount, 2}//,
		//{vertexCount, 3}
	};	

	HostDeviceArray<VeinNeighbors, gpuCount> neighbors {0, 1, 2}; // GPU_COUNT_DEPENDENT

	#else

	HostDeviceArray<cudaVec3, gpuCount> positions {{vertexCount, 0}};
	HostDeviceArray<cudaVec3, gpuCount> velocities {{vertexCount, 0}};
	HostDeviceArray<cudaVec3, gpuCount> forces {{vertexCount, 0}};

	HostDeviceArray<VeinNeighbors, gpuCount> neighbors {0};

	#endif

	HostDeviceArray<unsigned int*, gpuCount> indices;

	cudaVec3 centers{ triangleCount, veinGridGpu };

	VeinTriangles();
	VeinTriangles(const VeinTriangles& other);
	~VeinTriangles();

	/// <summary>
	/// device funtion to get triangle vertex absolute index
	/// </summary>
	__device__ inline unsigned int getIndex(int gpuId, int triangleIndex, VertexIndex vertexIndex) const
	{
		return indices[gpuId][3 * triangleIndex + vertexIndex];
	}

	void gatherForcesFromNeighbors(int gpuId, int gpuStart, int gpuEnd, int blocks, int threadsPerBlock);
	void propagateForcesIntoPositions(int blocks, int threadsPerBlock);

	void calculateCenters(int blocks, int threadsPerBlock);

#ifdef MULTI_GPU
	void broadcastPositionsAndVelocities(ncclComm_t* comms, cudaStream_t* streams);
	void reduceForces(ncclComm_t* comms, cudaStream_t* streams);
#endif

private:
	bool isCopy = false;

};