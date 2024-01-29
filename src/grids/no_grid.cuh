#pragma once

#include "base_grid.cuh"
#include "../objects/particles.cuh"

/// <summary>
/// Specialization of grid acting as no grid :)
/// </summary>
class NoGrid : public BaseGrid<NoGrid>
{
public:
	__host__ __device__ NoGrid() {}

	/// <summary>
	/// Recalculate grid basing on particles positions
	/// </summary>
	/// <param name="particles">simulation particles</param>
	/// <param name="objectCount">number of particles</param>
	inline void calculateGrid(const cudaVec3& particles, int objectCount) {}

	/// <summary>
	/// Calculate grid cell id from object position
	/// </summary>
	/// <param name="positions">object position</param>
	/// <returns>cell id</returns>
	inline int calculateCellId(float3 position) { return 0; }
};
