#pragma once

#include "../grids/no_grid.cuh"
#include "../grids/uniform_grid.cuh"
#include "../objects/blood_cells.cuh"
#include "../utilities/math.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


namespace sim
{
	/// <summary>
	/// Device method to detect and react on particle collisions
	/// </summary>
	/// <param name="bloodCells">data of blood cells</param>
	/// <param name="position1">first particle position</param>
	/// <param name="velocity1">first particle velocity</param>
	/// <param name="particleId1">first particle id</param>
	/// <param name="particleId2">second particle id</param>
	/// <param name="radius">first particle radius</param>
	/// <returns></returns>
	__device__ inline void detectCollision(BloodCells& bloodCells, float3 position1, float3 velocity1, int particleId1, int particleId2, float radius, float secondRadius)
	{
		float3 position2 = bloodCells.particles.positions.get(particleId2);
		float3 relativePosition = position1 - position2;
		float distanceSquared = length_squared(relativePosition);
		float minDistance = radius + secondRadius;

		if (distanceSquared <= minDistance*minDistance && distanceSquared >= 0.0001f)
		{
			float3 relativeVelocity = velocity1 - bloodCells.particles.velocities.get(particleId2);
			physics::addResilientForceOnCollision(relativePosition, relativeVelocity, distanceSquared, radius, particleId1, 0.2f, bloodCells.particles.forces);
		}
	}

	/// <summary>
	/// Detect Colllisions in all 27 cells unless some corner cases are present - specified by template parameters.
	/// </summary>
	/// <param name="bloodCells">data of blood cells</param>
	/// <param name="grid">particle uniform grid</param>
	/// <param name="p1">first particle position</param>
	/// <param name="v1">first particle velocity</param>
	/// <param name="particleId">particle id</param>
	/// <param name="cellId">grid cell id</param>
	/// <param name="boundingSpheresModel">data of bounding sphere in blood cell model</param>
	/// <param name="boundingModelIndex">blood cell model index shift for bounding sphere data</param>
	/// <returns></returns>
	template<int xMin, int xMax, int yMin, int yMax, int zMin, int zMax>
	__device__ void detectCollisionsInNeighborCells(BloodCells& bloodCells, UniformGrid& grid, float3 p1, float3 v1, int particleId, int cellId, float* boundingSpheresModel, int boundingModelIndex, int particlesInBloodCell, int bloodCellmodelStart, int particlesStart)
	{
		#pragma unroll
		for (int x = xMin; x <= xMax; x++)
		{
			#pragma unroll
			for (int y = yMin; y <= yMax; y++)
			{
				#pragma unroll
				for (int z = zMin; z <= zMax; z++)
				{
					int neighborCellId = cellId + z * grid.cellCountX * grid.cellCountY + y * grid.cellCountX + x;

					for (int i = grid.gridCellStarts[neighborCellId]; i <= grid.gridCellEnds[neighborCellId]; i++)
					{
						int secondParticleId = grid.particleIds[i];

						// TODO: Potential optimization - unroll the loops manually or think of a way to metaprogram the compiler to unroll
						// one particular iteration (0,0,0) differently than the others
						if (particleId == secondParticleId)
							continue;

						int secondModelIndex = bloodCellmodelStart + (secondParticleId - particlesStart) % particlesInBloodCell;
						// int modelIndex = bloodCellmodelStart + (particleId - particlesStart) % particlesInBloodCell;
						detectCollision(bloodCells, p1, v1, particleId, secondParticleId, boundingSpheresModel[boundingModelIndex], boundingSpheresModel[secondModelIndex]);
					}
				}
			}
		}
	}


	// Should have been a deleted function but CUDA doesn't like it
	template<typename T>
	__global__ void calculateParticleCollisions(BloodCells bloodCells, T grid, float* boundingSpheresModel,int particlesInBloodCell, int bloodCellmodelStart, int particlesStart) {}

	/// <summary>
	/// Calculate collisions between particles using UniformGrid
	/// </summary>
	/// <param name="bloodCells">data of blood cells</param>
	/// <param name="grid">particle uniform grid</param>
	/// <param name="boundingSpheresModel">data of bounding sphere in blood cell model</param>
	/// <param name="particlesInBloodCell">Number of particles in blood cell model</param>
	/// <param name="bloodCellmodelStart">index shift for blood cell model</param>
	/// <param name="particlesStart">index shift for particle data</param>
	/// <returns></returns>
	template<>
	__global__ void calculateParticleCollisions<UniformGrid>(BloodCells bloodCells, UniformGrid grid, float* boundingSpheresModel, int particlesInBloodCell, int bloodCellmodelStart, int particlesStart)
	{
		int id = particlesStart + blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= particleCount)
			return;

		int particleId = grid.particleIds[id];

		float3 p1 = bloodCells.particles.positions.get(particleId);
		float3 v1 = bloodCells.particles.velocities.get(particleId);

		int cellId = grid.gridCellIds[id];
		int xId = static_cast<int>((bloodCells.particles.positions.x[particleId] - minX) / grid.cellWidth);
		int yId = static_cast<int>((bloodCells.particles.positions.y[particleId] - minY) / grid.cellHeight);
		int zId = static_cast<int>((bloodCells.particles.positions.z[particleId] - minZ) / grid.cellDepth);
		
		// Check all corner cases and call the appropriate function specialization
		// Ugly but fast
		int boundingModelIndex = bloodCellmodelStart + (particleId - particlesStart) % particlesInBloodCell;
		if (xId < 1)
		{
			if (yId < 1)
			{
				if (zId < 1)
				{
					detectCollisionsInNeighborCells<0, 1, 0, 1, 0, 1>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
				else if (zId > grid.cellCountZ - 2)
				{
					detectCollisionsInNeighborCells<0, 1, 0, 1, -1, 0>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
				else
				{
					detectCollisionsInNeighborCells<0, 1, 0, 1, -1, 1>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
			}
			else if (yId > grid.cellCountY - 2)
			{
				if (zId < 1)
				{
					detectCollisionsInNeighborCells<0, 1, -1, 0, 0, 1>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
				else if (zId > grid.cellCountZ - 2)
				{
					detectCollisionsInNeighborCells<0, 1, -1, 0, -1, 0>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
				else
				{
					detectCollisionsInNeighborCells<0, 1, -1, 0, -1, 1>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
			}
			else
			{
				if (zId < 1)
				{
					detectCollisionsInNeighborCells<0, 1, -1, 1, 0, 1>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
				else if (zId > grid.cellCountZ - 2)
				{
					detectCollisionsInNeighborCells<0, 1, -1, 1, -1, 0>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
				else
				{
					detectCollisionsInNeighborCells<0, 1, -1, 1, -1, 1>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
			}
		}
		else if (xId > grid.cellCountX - 2)
		{
			if (yId < 1)
			{
				if (zId < 1)
				{
					detectCollisionsInNeighborCells<-1, 0, 0, 1, 0, 1>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
				else if (zId > grid.cellCountZ - 2)
				{
					detectCollisionsInNeighborCells<-1, 0, 0, 1, -1, 0>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
				else
				{
					detectCollisionsInNeighborCells<-1, 0, 0, 1, -1, 1>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
			}
			else if (yId > grid.cellCountY - 2)
			{
				if (zId < 1)
				{
					detectCollisionsInNeighborCells<-1, 0, -1, 0, 0, 1>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
				else if (zId > grid.cellCountZ - 2)
				{
					detectCollisionsInNeighborCells<-1, 0, -1, 0, -1, 0>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
				else
				{
					detectCollisionsInNeighborCells<-1, 0, -1, 0, -1, 1>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
			}
			else
			{
				if (zId < 1)
				{
					detectCollisionsInNeighborCells<-1, 0, -1, 1, 0, 1>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
				else if (zId > grid.cellCountZ - 2)
				{
					detectCollisionsInNeighborCells<-1, 0, -1, 1, -1, 0>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
				else
				{
					detectCollisionsInNeighborCells<-1, 0, -1, 1, -1, 1>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
			}
		}
		else
		{
			if (yId < 1)
			{
				if (zId < 1)
				{
					detectCollisionsInNeighborCells<-1, 1, 0, 1, 0, 1>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
				else if (zId > grid.cellCountZ - 2)
				{
					detectCollisionsInNeighborCells<-1, 1, 0, 1, -1, 0>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
				else
				{
					detectCollisionsInNeighborCells<-1, 1, 0, 1, -1, 1>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
			}
			else if (yId > grid.cellCountY - 2)
			{
				if (zId < 1)
				{
					detectCollisionsInNeighborCells<-1, 1, -1, 0, 0, 1>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
				else if (zId > grid.cellCountZ - 2)
				{
					detectCollisionsInNeighborCells<-1, 1, -1, 0, -1, 0>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
				else
				{
					detectCollisionsInNeighborCells<-1, 1, -1, 0, -1, 1>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
			}
			else
			{
				if (zId < 1)
				{
					detectCollisionsInNeighborCells<-1, 1, -1, 1, 0, 1>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
				else if (zId > grid.cellCountZ - 2)
				{
					detectCollisionsInNeighborCells<-1, 1, -1, 1, -1, 0>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
				else
				{
					detectCollisionsInNeighborCells<-1, 1, -1, 1, -1, 1>(bloodCells, grid, p1, v1, particleId, cellId, boundingSpheresModel, boundingModelIndex, particlesInBloodCell, bloodCellmodelStart, particlesStart);
				}
			}
		}
	}


	/// <summary>
	/// Calculate collisions between particles without any grid (naive implementation)
	/// </summary>
	/// <param name="bloodCells">data of blood cells</param>
	/// <param name="grid">particle uniform grid</param>
	/// <param name="boundingSpheresModel">data of bounding sphere in blood cell model</param>
	/// <param name="particlesInBloodCell">Number of particles in blood cell model</param>
	/// <param name="bloodCellmodelStart">index shift for blood cell model</param>
	/// <param name="particlesStart">index shift for particle data</param>
	/// <returns></returns>
	 template<>
	 __global__ void calculateParticleCollisions<NoGrid>(BloodCells bloodCells, NoGrid grid, float* boundingSpheresModel, int particlesInBloodCell, int bloodCellmodelStart, int particlesStart)
	 {
	 	int id = blockIdx.x * blockDim.x + threadIdx.x;
	 	if (id >= particleCount)
	 		return;

	 	float3 p1 = bloodCells.particles.positions.get(id);
	 	float3 v1 = bloodCells.particles.positions.get(id);

	 	// Naive implementation
	 	for (int i = 0; i < particleCount; i++)
	 	{
	 		if (id == i)
	 			continue;

	 		// if to use in the future, last arguments should be changed
	 		detectCollision(bloodCells, p1, v1, id, i, 0.0f, 0.0f);
	 	}
	 }
}