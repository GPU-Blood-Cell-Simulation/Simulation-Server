#pragma once

#include "base_grid.cuh"
#include "../objects/particles.cuh"

/// <summary>
/// Represents a grid of uniform cell size
/// </summary>
class UniformGrid : public BaseGrid<UniformGrid>
{
private:
	bool isCopy = false;

public:

	int cellWidth;
	int cellHeight;
	int cellDepth;
	int cellCountX;
	int cellCountY;
	int cellCountZ;
	int cellAmount;
	int objectCount;

	int* gridCellIds = 0;
	int* particleIds = 0;
	int* gridCellStarts = 0;
	int* gridCellEnds = 0;

	UniformGrid(const int objectCount, int cellWidth, int cellHeight, int cellDepth);
	UniformGrid(const UniformGrid& other);
	~UniformGrid();

	inline void calculateGrid(const Particles& particles, int objectCount)
	{
		calculateGrid(particles.positions.x, particles.positions.y, particles.positions.z, objectCount);
	}

	void calculateGrid(const float* positionX, const float* positionY, const float* positionZ, int objectCount);

	__device__ int calculateCellId(float3 position);
};
