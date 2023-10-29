#pragma once

#include "../grids/uniform_grid.cuh"
#include "../grids/no_grid.cuh"
#include "../objects/blood_cells.cuh"
#include "../objects/vein_triangles.cuh"
#include "../utilities/math.cuh"

#include <cmath>
#include <variant>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>


using Grid = std::variant<UniformGrid*, NoGrid*>;

namespace sim
{
	// a helper struct for calculating triangle intersections
	struct ray
	{
		float3 origin;
		float3 direction;
		float t = 1e10f;

		// rays may be used to determine intersection with objects
		// so its easy to store object index inside ray
		int objectIndex = 0;

		__device__ ray(float3 origin, float3 direction);
	};

	__device__ float3 calculateBaricentric(float3 point, float3 v1, float3 v2, float3 v3);

	#pragma region Main Collision Template Kernels
	template<typename T>
	__global__ void detectVeinCollisionsAndPropagateParticles(BloodCells cells, VeinTriangles triangles, T triangleGrid) {}

	template<>
	__global__ void detectVeinCollisionsAndPropagateParticles<NoGrid>(BloodCells cells, VeinTriangles triangles, NoGrid triangleGrid);

	template<>
	__global__ void detectVeinCollisionsAndPropagateParticles<UniformGrid>(BloodCells cells, VeinTriangles triangles, UniformGrid triangleGrid);
	#pragma endregion

	template<int xMin, int xMax, int yMin, int yMax, int zMin, int zMax>
	__device__ bool calculateSideCollisions(float3 position, ray& r, float3& reflectionVector, VeinTriangles& triangles, UniformGrid& triangleGrid);


	__device__ bool realCollisionDetection(float3 v0, float3 v1, float3 v2, ray& r, float3& reflectionVector);

	__device__ float3 calculateBaricentric(float3 point, float3 v0, float3 v1, float3 v2);

	__device__ bool modifyVelocityIfPositionOutOfBounds(float3& position, float3& velocity, float3 velocityNormalized);
}
