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
	inline void calculateGrid(const Particles& particles, int objectCount) {}

	/// <summary>
	/// Recalculate grid basing on objects positions
	/// </summary>
	/// <param name="positionX">device buffer of X's of positions</param>
	/// <param name="positionY">device buffer of Y's of positions</param>
	/// <param name="positionZ">device buffer of Z's of positions</param>
	/// <param name="objectCount">number of objects</param>
	inline void calculateGrid(const float* positionX, const float* positionY, const float* positionZ, int objectCount) {}

	/// <summary>
	/// Calculate grid cell id from object position
	/// </summary>
	/// <param name="positions">object position</param>
	/// <returns>cell id</returns>
	inline int calculateCellId(float3 position) { return 0; }
};
