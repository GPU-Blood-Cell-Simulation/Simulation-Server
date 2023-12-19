#include "uniform_grid.cuh"

#include "../config/simulation.hpp"
#include "../meta_factory/vein_factory.hpp"
#include "../objects/particles.cuh"
#include "../utilities/cuda_handle_error.cuh"

#include <cstdio>
#include <cstdlib>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define max(a,b) ( a > b ? a : b)
#define min(a,b) ( a > b ? b : a)


__device__ int calculateIdForCell(float x, float y, float z, int cellWidth, int cellHeight, int cellDepth)
{
	if (x < minX || x > maxX || y < minY || y > maxY || z < minZ || z > maxZ)
	{
		printf("Position out of grid bounds: (%f, %f, %f)\n", x, y, z);
	}

	// should we clamp x,y,z if out of bounds?
	return
		static_cast<int>(min(maxZ - minZ, max(0, (z - minZ) / cellDepth))) * static_cast<int>(width / cellWidth) * static_cast<int>(height / cellHeight) +
		static_cast<int>(min(maxY - minY, max(0, (y - minY) / cellHeight))) * static_cast<int>(width / cellWidth) +
		static_cast<int>(min(maxX - minX, max(0, (x - minX) / cellWidth)));
}

__global__ void calculateCellIdKernel(const float* positionX, const float* positionY, const float* positionZ,
	int* cellIds, int* particleIds, int particleCount, int cellWidth, int cellHeight, int cellDepth)
{
	int particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId >= particleCount)
		return;

	int cellId = calculateIdForCell(positionX[particleId], positionY[particleId], positionZ[particleId], cellWidth, cellHeight, cellDepth);

	particleIds[particleId] = particleId;
	cellIds[particleId] = cellId;
}

__global__ void calculateStartAndEndOfCellKernel(const float* positionX, const float* positionY, const float* positionZ,
	const int* cellIds, const int* particleIds, int* cellStarts, int* cellEnds, int particleCount)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= particleCount)
		return;

	int currentCellId = cellIds[id];

	// Check if the previous cell id was different - it would mean we found the start of a cell
	if (id > 0 && currentCellId != cellIds[id - 1])
	{
		cellStarts[currentCellId] = id;
	}

	// Check if the next cell id was different - it would mean we found the end of a cell
	if (id < particleCount - 1 && currentCellId != cellIds[id + 1])
	{
		cellEnds[currentCellId] = id;
	}

	if (id == 0)
	{
		cellStarts[cellIds[0]] = 0;
	}
	if (id == particleCount - 1)
	{
		cellStarts[cellIds[particleCount - 1]] = particleCount - 1;
	}
}

// Allocate GPU buffers for the index buffers
UniformGrid::UniformGrid(const int objectCount, int cellWidth, int cellHeight, int cellDepth) :
	objectCount(objectCount), cellWidth(cellWidth), cellHeight(cellHeight), cellDepth(cellDepth),
	cellCountX(static_cast<int>(std::ceil(width / cellWidth) + 0.5f)),
	cellCountY(static_cast<int>(std::ceil(height / cellHeight) + 0.5f)),
	cellCountZ(static_cast<int>(std::ceil(depth / cellDepth) + 0.5f)),
	cellCount(cellCountX * cellCountY * cellCountZ)
{
	HANDLE_ERROR(cudaMalloc((void**)&gridCellIds, objectCount * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&particleIds, objectCount * sizeof(int)));

	HANDLE_ERROR(cudaMalloc((void**)&gridCellStarts, cellCount * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&gridCellEnds, cellCount * sizeof(int)));
}

UniformGrid::UniformGrid(const UniformGrid& other) : isCopy(true), gridCellIds(other.gridCellIds), particleIds(other.particleIds),
	gridCellStarts(other.gridCellStarts), gridCellEnds(other.gridCellEnds),
	objectCount(other.objectCount), cellWidth(other.cellWidth), cellHeight(other.cellHeight), cellDepth(other.cellDepth),
	cellCountX(other.cellCountX),
	cellCountY(other.cellCountY),
	cellCountZ(other.cellCountZ),
	cellCount(other.cellCount)
{}

UniformGrid::~UniformGrid()
{
	if (!isCopy)
	{
		HANDLE_ERROR(cudaFree(gridCellIds));
		HANDLE_ERROR(cudaFree(particleIds));
		HANDLE_ERROR(cudaFree(gridCellStarts));
		HANDLE_ERROR(cudaFree(gridCellEnds));
	}
}

void UniformGrid::calculateGrid(const float* positionX, const float* positionY, const float* positionZ, int objectCount)
{
	// Calculate launch parameters

	const int threadsPerBlock = objectCount > 1024 ? 1024 : objectCount;
	const int blocks = (objectCount + threadsPerBlock - 1) / threadsPerBlock;

	// 1. Calculate cell id for every particle and store as pair (cell id, particle id) in two buffers
	calculateCellIdKernel << <blocks, threadsPerBlock >> >
		(positionX, positionY, positionZ, gridCellIds, particleIds, objectCount, cellWidth, cellHeight, cellDepth);

	// 2. Sort particle ids by cell id

	thrust::device_ptr<int> keys = thrust::device_pointer_cast<int>(gridCellIds);
	thrust::device_ptr<int> values = thrust::device_pointer_cast<int>(particleIds);

	thrust::stable_sort_by_key(keys, keys + objectCount, values);

	// 3. Find the start and end of every cell

	calculateStartAndEndOfCellKernel << <blocks, threadsPerBlock >> >
		(positionX, positionY, positionZ, gridCellIds, particleIds, gridCellStarts, gridCellEnds, objectCount);
}

__device__ int UniformGrid::calculateCellId(float3 position)
{
	return calculateIdForCell(position.x, position.y, position.z, cellWidth, cellHeight, cellDepth);
}