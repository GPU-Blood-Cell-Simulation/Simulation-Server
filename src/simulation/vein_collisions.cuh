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

	__device__ bool realCollisionDetection(float3 v0, float3 v1, float3 v2, ray& r, float3& reflectionVector);

	__device__ float3 calculateBaricentric(float3 point, float3 v0, float3 v1, float3 v2);

	__device__ bool modifyVelocityIfPositionOutOfBounds(float3& position, float3 newPosition, float3& velocity, float3 normalizedVelocity);

	template<int xMin, int xMax, int yMin, int yMax, int zMin, int zMax>
	__device__ bool calculateSideCollisions(float3 position, ray& r, float3& reflectionVector, VeinTriangles& triangles, UniformGrid& triangleGrid)
	{
		unsigned int cellId = triangleGrid.calculateCellId(position);

#pragma unroll
		for (int x = xMin; x <= xMax; x++)
		{
#pragma unroll	
			for (int y = yMin; y <= yMax; y++)
			{
#pragma unroll
				for (int z = zMin; z <= zMax; z++)
				{
					int neighborCellId = cellId + z * triangleGrid.cellCountX * triangleGrid.cellCountY + y * triangleGrid.cellCountX + x;
					for (int i = triangleGrid.gridCellStarts[neighborCellId]; i <= triangleGrid.gridCellEnds[neighborCellId]; i++)
					{
						// triangle vectices and edges
						unsigned int triangleId = triangleGrid.particleIds[i];
						float3 v0 = triangles.positions.get(triangles.getIndex(triangleId, vertex0));
						float3 v1 = triangles.positions.get(triangles.getIndex(triangleId, vertex1));
						float3 v2 = triangles.positions.get(triangles.getIndex(triangleId, vertex2));

						if (!realCollisionDetection(v0, v1, v2, r, reflectionVector))
							continue;

						r.objectIndex = triangleId;
						return true;
					}
				}
			}
		}
		return false;
	}

#pragma region Main Collision Template Kernels
	 template<typename T>
	 __global__ void detectVeinCollisionsAndPropagateForces(BloodCells bloodCells, VeinTriangles triangles, T triangleGrid, float* boundingSpheresModel,
		 int particlesInBloodCell, int bloodCellmodelStart, int particlesStart) {}

	template<>
	 __global__ void detectVeinCollisionsAndPropagateForces<NoGrid>(BloodCells bloodCells, VeinTriangles triangles, NoGrid triangleGrid, float* boundingSpheresModel,
		 int particlesInBloodCell, int bloodCellmodelStart, int particlesStart);

	template<>
	__global__ void detectVeinCollisionsAndPropagateForces<UniformGrid>(BloodCells bloodCells, VeinTriangles triangles, UniformGrid triangleGrid, float* boundingSpheresModel,
		int particlesInBloodCell, int bloodCellmodelStart, int particlesStart);

	
#pragma endregion

}
