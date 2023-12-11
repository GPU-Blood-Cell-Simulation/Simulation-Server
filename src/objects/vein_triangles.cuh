#pragma once

#include "../simulation/physics.cuh"
#include "../meta_factory/vein_factory.hpp"
#include "../utilities/cuda_vec3.cuh"
#include "../utilities/math.cuh"
#include "../utilities/vertex_index_enum.hpp"

#include <vector>
#include <tuple>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/// <summary>
/// Vein triangles
/// </summary>
class VeinTriangles
{
public:
	const unsigned int triangleCount;
	const unsigned int vertexCount;

	cudaVec3 positions;
	cudaVec3 velocities;
	cudaVec3 forces;

	unsigned int* indices;
	cudaVec3 centers;

	// TODO: move these out of the class
	const float veinVertexHorizontalDistance;
	const float veinVertexNonHorizontalDistances[3];

	VeinTriangles(const std::vector<glm::vec3>& positions, const std::vector<unsigned int> &indices, const std::tuple<float, float, float>& springLengths);
	VeinTriangles(const VeinTriangles& other);
	~VeinTriangles();


	__device__ inline unsigned int getIndex(int triangleIndex, VertexIndex vertexIndex) const
	{
		return indices[3 * triangleIndex + vertexIndex];
	}

	void gatherForcesFromNeighbors(int blocks, int threadsPerBlock);
	void propagateForcesIntoPositions(int blocks, int threadsPerBlock);

	void calculateCenters(int blocks, int threadsPerBlock);

private:
	bool isCopy = false;

	/// !!! STILL NOT IMPLEMENTED !!!
	/// <param name="vertexIndex">0, 1 or 2 as triangle vertices</param>
	/// <returns></returns>
	__device__ inline void atomicAdd(int triangleIndex, VertexIndex vertexIndex, float3 value)
	{
		int index = indices[3 * triangleIndex + vertexIndex];
		positions.atomicAddVec3(index, value);
	}
};