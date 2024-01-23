#pragma once

#include "base_grid.cuh"
#include "../objects/particles.cuh"
#include "../utilities/host_device_array.cuh"

#ifdef MULTI_GPU
#include "../utilities/nccl_operations.cuh"
#endif


/// <summary>
/// Represents a grid of uniform cell size
/// </summary>
class UniformGrid : public BaseGrid<UniformGrid>
{
private:
	bool isCopy = false;
	int gpuId;

public:

	int cellWidth;
	int cellHeight;
	int cellDepth;
	int cellCountX;
	int cellCountY;
	int cellCountZ;
	int cellCount;
	int objectCount;

	HostDeviceArray<int*, gpuCount> gridCellIds;
	HostDeviceArray<int*, gpuCount> particleIds;
	HostDeviceArray<int*, gpuCount> gridCellStarts;
	HostDeviceArray<int*, gpuCount> gridCellEnds;

	UniformGrid(int gpuId, int objectCount, int cellWidth, int cellHeight, int cellDepth);
	UniformGrid(const UniformGrid& other);
	~UniformGrid();

	/// <summary>
	/// Recalculate grid basing on particles positions
	/// </summary>
	/// <param name="particles">simulation particles</param>
	/// <param name="objectCount">number of particles</param>
	void calculateGrid(const cudaVec3& positions, int objectCount);

	/// <summary>
	/// Calculate grid cell id from object position
	/// </summary>
	/// <param name="positions">object position</param>
	/// <returns>cell id</returns>
	__device__ int calculateCellId(float3 position);

	#ifdef MULTI_GPU
	void broadcastGrid(ncclComm_t* comms, cudaStream_t* streams);
	#endif
};
