#pragma once

#include "base_grid.cuh"
#include "../objects/particles.cuh"
#include "../utilities/host_device_array.cuh"

#ifdef MULTI_GPU
#include <nccl.h>
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
	inline void calculateGrid(const Particles& particles, int objectCount)
	{
		calculateGrid(particles.positions.x, particles.positions.y, particles.positions.z, objectCount);
	}

	/// <summary>
	/// Recalculate grid basing on objects positions
	/// </summary>
	/// <param name="positionX">device buffer of X's of positions</param>
	/// <param name="positionY">device buffer of Y's of positions</param>
	/// <param name="positionZ">device buffer of Z's of positions</param>
	/// <param name="objectCount">number of objects</param>
	void calculateGrid(const float* positionX, const float* positionY, const float* positionZ, int objectCount);

	/// <summary>
	/// Calculate grid cell id from object position
	/// </summary>
	/// <param name="positions">object position</param>
	/// <returns>cell id</returns>
	__device__ int calculateCellId(float3 position);

	void broadcast(ncclComm_t* comms, cudaStream_t* streams);
};
