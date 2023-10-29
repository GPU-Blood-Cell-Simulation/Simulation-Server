#pragma once

#include "base_grid.cuh"
#include "../objects/particles.cuh"

class NoGrid : public BaseGrid<NoGrid>
{
public:
	__host__ __device__ NoGrid() {}

	inline void calculateGrid(const Particles& particles, int objectCount) {}

	inline void calculateGrid(const float* positionX, const float* positionY, const float* positionZ, int objectCount) {}

	inline int calculateCellId(float3 position) {}
};