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
	/// Executes triangle collision check in sorrounding grid cells
	/// </summary>
	template<int xMin, int xMax, int yMin, int yMax, int zMin, int zMax>
	__device__ bool calculateSideCollisions(int gpuId, float3 position, ray& r, float3& reflectionVector, VeinTriangles& triangles, UniformGrid& triangleGrid)
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
					for (int i = triangleGrid.gridCellStarts[gpuId][neighborCellId]; i <= triangleGrid.gridCellEnds[gpuId][neighborCellId]; i++)
					{
						// triangle vectices and edges
						unsigned int triangleId = triangleGrid.particleIds[gpuId][i];
						float3 v0 = triangles.positions[gpuId].get(triangles.getIndex(gpuId, triangleId, vertex0));
						float3 v1 = triangles.positions[gpuId].get(triangles.getIndex(gpuId, triangleId, vertex1));
						float3 v2 = triangles.positions[gpuId].get(triangles.getIndex(gpuId, triangleId, vertex2));

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
	__global__ void detectVeinCollisions(int gpuId, int gpuStart, int gpuEnd, BloodCells bloodCells, VeinTriangles triangles, T triangleGrid, float* boundingSpheresModel,
		int bloodCellsOfType, int particlesInBloodCell, int bloodCellmodelStart, int particlesStart) {}

	/// <summary>
	/// Main kernel to detect collisions with triangles
	/// </summary>
	/// <param name="gpuId">the gpu on which the kernel is launched</param>
	/// <param name="gpuStart">start of the vein vertex array range for this gpu</param>
	/// <param name="gpuEnd">end of the vein vertex array range for this gpu</param>
	/// <param name="bloodCells">blood cell device data</param>
	/// <param name="triangles">triangles device data</param>
	/// <param name="triangleGrid">triangle grid</param>
	/// <param name="boundingSpheresModel">data of bounding sphere in blood cell model</param>
	/// <param name="particlesInBloodCell">Number of particles in blood cell model</param>
	/// <param name="bloodCellmodelStart">index shift for blood cell model</param>
	/// <param name="particlesStart">index shift for particle data</param>
	/// <returns></returns>
	template<>
	__global__ void detectVeinCollisions<NoGrid>(int gpuId, int gpuStart, int gpuEnd, BloodCells bloodCells, VeinTriangles triangles, NoGrid triangleGrid, float* boundingSpheresModel,
		int bloodCellsOfType, int particlesInBloodCell, int bloodCellmodelStart, int particlesStart);

	/// <summary>
	/// Main kernel to detect collisions with triangles
	/// </summary>
	/// <param name="gpuId">the gpu on which the kernel is launched</param>
	/// <param name="gpuStart">start of the vein vertex array range for this gpu</param>
	/// <param name="gpuEnd">end of the vein vertex array range for this gpu</param>
	/// <param name="bloodCells">blood cell device data</param>
	/// <param name="triangles">triangles device data</param>
	/// <param name="triangleGrid">triangle grid</param>
	/// <param name="boundingSpheresModel">data of bounding sphere in blood cell model</param>
	/// <param name="particlesInBloodCell">Number of particles in blood cell model</param>
	/// <param name="bloodCellmodelStart">index shift for blood cell model</param>
	/// <param name="particlesStart">index shift for particle data</param>
	/// <returns></returns>
	template<>
	__global__ void detectVeinCollisions<UniformGrid>(int gpuId, int gpuStart, int gpuEnd, BloodCells bloodCells, VeinTriangles triangles, UniformGrid triangleGrid, float* boundingSpheresModel,
		int bloodCellsOfType, int particlesInBloodCell, int bloodCellmodelStart, int particlesStart);

}
