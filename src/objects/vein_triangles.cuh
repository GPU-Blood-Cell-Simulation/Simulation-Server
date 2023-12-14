#pragma once

#include "../config/physics.hpp"
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
	const int triangleCount = veinIndexCount / 3;
	const int vertexCount = veinPositionCount;

	cudaVec3 positions{ vertexCount };
	cudaVec3 velocities{ vertexCount };
	cudaVec3 forces{ vertexCount };

	unsigned int* indices;
	cudaVec3 centers{ triangleCount };

	// TODO: move these out of the class
	const float veinVertexHorizontalDistance;
	const float veinVertexNonHorizontalDistances[3];

	VeinTriangles(const std::tuple<float, float, float>& springLengths);
	VeinTriangles(const VeinTriangles& other);
	~VeinTriangles();


	__device__ inline unsigned int getIndex(int triangleIndex, VertexIndex vertexIndex) const
	{
		return indices[3 * triangleIndex + vertexIndex];
	}

	__device__ inline float calculateVeinSpringForce(float3 p1, float3 p2, float3 v1, float3 v2, float springLength)
	{
		return (length(p1 - p2) - springLength) * vein_k_sniff + dot(normalize(p1 - p2), (v1 - v2)) * vein_d_fact;
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