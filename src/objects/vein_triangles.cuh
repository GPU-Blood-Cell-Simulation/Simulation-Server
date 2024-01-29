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
		{vertexCount, 2},
		{vertexCount, 3}
	};
	HostDeviceArray<cudaVec3, gpuCount> velocities { // GPU_COUNT_DEPENDENT
		{vertexCount, 0},
		{vertexCount, 1},
		{vertexCount, 2},
		{vertexCount, 3}
	};
	HostDeviceArray<cudaVec3, gpuCount> forces{ // GPU_COUNT_DEPENDENT
		{vertexCount, 0},
		{vertexCount, 1},
		{vertexCount, 2},
		{vertexCount, 3}
	};	

	HostDeviceArray<VeinNeighbors, gpuCount> neighbors {0, 1, 2, 3}; // GPU_COUNT_DEPENDENT

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
	/// Device funtion to get triangle vertex absolute index
	/// </summary>
	/// <param name="gpuId">The gpu on which the function is called</param>
	/// <param name="triangleIndex">Index of the triangle</param>
	/// <param name="vertexIndex">Index of the vertex in the triangle (0,1,2)</param>
	/// <returns>The real index of the vertex in the vertex array</returns>
	__device__ inline unsigned int getIndex(int gpuId, int triangleIndex, VertexIndex vertexIndex) const
	{
		return indices[gpuId][3 * triangleIndex + vertexIndex];
	}

	/// <summary>
	/// Adjust the force acting on particles based on the neighbors attached to them by springs
	/// </summary>
	/// <param name="gpuId">The gpu on which the kernel is launched</param>
	/// <param name="gpuStart">Start of the vein vertex array for this gpu</param>
	/// <param name="gpuEnd">End of the vein vertex  for this gpu</param>
	/// <param name="blocks">CUDA block count</param>
	/// <param name="threadsPerBlock">CUDA threads in a single block</param>
	void gatherForcesFromNeighbors(int gpuId, int gpuStart, int gpuEnd, int blocks, int threadsPerBlock);

	/// <summary>
	/// Propagate forces into velocities and velocities into positions
	/// </summary>
	/// <param name="blocks">CUDA block count</param>
	/// <param name="threadsPerBlock">CUDA threads in a single block</param>
	void propagateForcesIntoPositions(int blocks, int threadsPerBlock);

	/// <summary>
	/// Calculate the centers of each triangle (necessary for calculating the grid)
	/// </summary>
	/// <param name="blocks">CUDA block count</param>
	/// <param name="threadsPerBlock">CUDA threads in a single block</param>
	void calculateCenters(int blocks, int threadsPerBlock);

#ifdef MULTI_GPU
	/// <summary>
	/// Broadcast vein data from the root gpu to all others
	/// </summary>
	/// <param name="blocks">NCCL comm array</param>
	/// <param name="blocks">NCCL synchronization stream array</param>
	void broadcastPositionsAndVelocities(ncclComm_t* comms, cudaStream_t* streams);

	/// <summary>
	/// Reduce vein forces from all gpus to the root node
	/// </summary>
	/// <param name="blocks">NCCL comm array</param>
	/// <param name="blocks">NCCL synchronization stream array</param>
	void reduceForces(ncclComm_t* comms, cudaStream_t* streams);
#endif

private:
	bool isCopy = false;

};