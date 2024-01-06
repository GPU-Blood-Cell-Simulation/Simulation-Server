#pragma once

#include "../grids/grid_definition.hpp"
#include "../objects/blood_cells.cuh"
#include "../objects/vein_triangles.cuh"
#include "../utilities/math.cuh"

#include <cmath>
#include <variant>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


namespace sim
{
	/// <summary>
	/// a helper struct for calculating triangle intersections
	/// </summary>
	struct ray
	{
		float3 origin;
		float3 direction;
		float3 normal;
		float t = 1e10f;

		// rays may be used to determine intersection with objects
		// so its easy to store object index inside ray
		int objectIndex = 0;

		__device__ ray(float3 origin, float3 direction);
	};

	/// <summary>
	/// Calculates baricentric coordinate for point in triangle 
	/// </summary>
	/// <param name="point">input point</param>
	/// <param name="v1">vertex 1</param>
	/// <param name="v2">vertex 2</param>
	/// <param name="v3">vertex 3</param>
	/// <returns>baricentric coords of point</returns>
	__device__ float3 calculateBaricentric(float3 point, float3 v1, float3 v2, float3 v3);

	/// <summary>
	/// Executes algorith to determine if ray crosses triangle
	/// </summary>
	/// <param name="v0">vertex 0</param>
	/// <param name="v1">vertex 1</param>
	/// <param name="v2">vertex 2</param>
	/// <param name="r">ray</param>
	/// <param name="reflectionVector">reference to output reflection vector</param>
	/// <returns>if collision occured</returns>
	__device__ bool realCollisionDetection(float3 v0, float3 v1, float3 v2, ray& r, float3& reflectionVector);

	__device__ float3 calculateBaricentric(float3 point, float3 v0, float3 v1, float3 v2);

	/// <summary>
	/// Handles situation of position out of defined bounds
	/// </summary>
	/// <param name="position">previous position</param>
	/// <param name="newPosition">new position</param>
	/// <param name="velocity">particle velocity</param>
	/// <param name="normalizedVelocity">normalized particle velocity</param>
	/// <returns>if new position was out of bound</returns>
	__device__ bool modifyVelocityIfPositionOutOfBounds(float3& position, float3 newPosition, float3& velocity, float3 normalizedVelocity);

	/// <summary>
	/// Executes triangle collision check in sorrounding grid cells
	/// </summary>
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

	 template<typename T>
	 __global__ void detectVeinCollisionsAndPropagateForces(BloodCells bloodCells, VeinTriangles triangles, T triangleGrid, float* boundingSpheresModel,
		 int particlesInBloodCell, int bloodCellmodelStart, int particlesStart) {}

	/// <summary>
	/// Main kernel to detect collisions with triangles
	/// </summary>
	/// <param name="bloodCells">blood cell device data</param>
	/// <param name="triangles">triangles device data</param>
	/// <param name="triangleGrid">triangle grid</param>
	/// <param name="boundingSpheresModel">data of bounding sphere in blood cell model</param>
	/// <param name="particlesInBloodCell">Number of particles in blood cell model</param>
	/// <param name="bloodCellmodelStart">index shift for blood cell model</param>
	/// <param name="particlesStart">index shift for particle data</param>
	/// <returns></returns>
	template<>
	 __global__ void detectVeinCollisionsAndPropagateForces<NoGrid>(BloodCells bloodCells, VeinTriangles triangles, NoGrid triangleGrid, float* boundingSpheresModel,
		 int particlesInBloodCell, int bloodCellmodelStart, int particlesStart);

	 /// <summary>
	 /// Main kernel to detect collisions with triangles
	 /// </summary>
	 /// <param name="bloodCells">blood cell device data</param>
	 /// <param name="triangles">triangles device data</param>
	 /// <param name="triangleGrid">triangle grid</param>
	 /// <param name="boundingSpheresModel">data of bounding sphere in blood cell model</param>
	 /// <param name="particlesInBloodCell">Number of particles in blood cell model</param>
	 /// <param name="bloodCellmodelStart">index shift for blood cell model</param>
	 /// <param name="particlesStart">index shift for particle data</param>
	 /// <returns></returns>
	template<>
	__global__ void detectVeinCollisionsAndPropagateForces<UniformGrid>(BloodCells bloodCells, VeinTriangles triangles, UniformGrid triangleGrid, float* boundingSpheresModel,
		int particlesInBloodCell, int bloodCellmodelStart, int particlesStart);

}
