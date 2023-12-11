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

	__device__ bool realCollisionDetection(float3 v0, float3 v1, float3 v2, ray& r, float3& reflectionVector);

	__device__ float3 calculateBaricentric(float3 point, float3 v0, float3 v1, float3 v2);

	__device__ bool modifyVelocityIfPositionOutOfBounds(float3& position, float3& velocity, float3 velocityNormalized);

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
	template<typename T, int particlesInBloodCell, int bloodCellmodelStart, int particlesStart>
	__global__ void detectVeinCollisionsAndPropagateForces(BloodCells bloodCells, VeinTriangles triangles, T triangleGrid, float* boundingSpheresModel) {}

	template<int particlesInBloodCell, int bloodCellmodelStart, int particlesStart>
	__global__ void detectVeinCollisionsAndPropagateForces<NoGrid>(BloodCells bloodCells, VeinTriangles triangles, NoGrid triangleGrid, float* boundingSpheresModel)
	{
		{
			int particleId = blockDim.x * blockIdx.x + threadIdx.x;

			if (particleId >= particleCount)
				return;

			// propagate force into velocities
			float3 F = bloodCells.particles.forces.get(particleId);
			float3 velocity = bloodCells.particles.velocities.get(particleId);
			float3 initialVelocity = velocity;
			float3 pos = bloodCells.particles.positions.get(particleId);

			velocity = velocity + dt * F;
			float3 velocityDir = normalize(velocity);


			// cubical bounds
			if (modifyVelocityIfPositionOutOfBounds(pos, velocity, velocityDir)) {
				goto set_particle_values;
			}

			ray r(pos, velocityDir);
			float3 reflectedVelociy = make_float3(0, 0, 0);

			bool collisionOccured = false;
			for (int triangleId = 0; triangleId < triangles.triangleCount; ++triangleId)
			{
				constexpr float EPS = 1e-7f;
				// triangle vectices and edges
				float3 v0 = triangles.positions.get(triangles.getIndex(triangleId, vertex0));
				float3 v1 = triangles.positions.get(triangles.getIndex(triangleId, vertex1));
				float3 v2 = triangles.positions.get(triangles.getIndex(triangleId, vertex2));
				const float3 edge1 = v1 - v0;
				const float3 edge2 = v2 - v0;

				const float3 h = cross(r.direction, edge2);
				const float a = dot(edge1, h);
				if (a > -EPS && a < EPS)
					continue; // ray parallel to triangle

				const float f = 1 / a;
				const float3 s = r.origin - v0;
				const float u = f * dot(s, h);
				if (u < 0 || u > 1)
					continue;
				if (!realCollisionDetection(v0, v1, v2, r, reflectedVelociy))
					continue;

				r.objectIndex = triangleId;
				collisionOccured = true;
				break;
			}

			float3 relativePosition = pos - (pos + r.t * r.direction);
			float distanceSquared = length_squared(relativePosition);
			
			if (collisionOccured && distanceSquared <= veinImpactDistance * veinImpactDistance)
			{
				// handle particle on collision
				if (distanceSquared > veinImpactMinimalForceDistance * veinImpactMinimalForceDistance)
				{
					physics::addResilientForceOnCollision(relativePosition, velocity, distanceSquared, particleId,
						boundingSpheresModel[bloodCellmodelStart + (particleId - particlesStart) % particlesInBloodCell], 0.5f, bloodCells.particles.forces);
				}
				float speed = length(velocity);
				velocity = velocityCollisionDamping * speed * reflectedVelociy;
				bloodCells.particles.velocities.set(particleId, velocity);

				// handle vein on collision
				float3 ds = 0.8f * velocityDir;
				unsigned int vertexIndex0 = triangles.getIndex(r.objectIndex, vertex0);
				unsigned int vertexIndex1 = triangles.getIndex(r.objectIndex, vertex1);
				unsigned int vertexIndex2 = triangles.getIndex(r.objectIndex, vertex2);

				float3 v0 = triangles.positions.get(vertexIndex0);
				float3 v1 = triangles.positions.get(vertexIndex1);
				float3 v2 = triangles.positions.get(vertexIndex2);

				float3 baricentric = calculateBaricentric(pos + r.t * r.direction, v0, v1, v2);

				// TODO:
				// Can these lines generate concurrent write conflicts? Unlikely but not impossible. Think about it. - Filip
				// Here we probably should use atomicAdd. - Hubert
				// move triangle a bit
				triangles.forces.add(vertexIndex0, baricentric.x * ds);
				triangles.forces.add(vertexIndex1, baricentric.y * ds);
				triangles.forces.add(vertexIndex2, baricentric.z * ds);
			}

		set_particle_values:

			physics::propagateForcesInParticles(particleId, bloodCells, velocity, initialVelocity);
		}
	}

	template<int particlesInBloodCell, int bloodCellmodelStart, int particlesStart>
	__global__ void detectVeinCollisionsAndPropagateForces<UniformGrid>(BloodCells bloodCells, VeinTriangles triangles, UniformGrid triangleGrid, float* boundingSpheresModel)
	{
		int particleId = blockDim.x * blockIdx.x + threadIdx.x;

		if (particleId >= particleCount)
			return;

		float3 velocity = bloodCells.particles.velocities.get(particleId);
		float3 initialVelocity = velocity;
		float3 pos = bloodCells.particles.positions.get(particleId);

		// TEST
		//float3 velocity = velocity + float3{ 0, 0.1f , 0 };
		//return;


		// TODO: is there a faster way to calculate this?
		/*if (velocity.x != 0 && velocity.y != 0 && velocity.z != 0)
			goto set_particle_values;*/

		float3 velocityDir = normalize(velocity);

		// cubical bounds
		if (modifyVelocityIfPositionOutOfBounds(pos, velocity, velocityDir)) {
			goto set_particle_values;
		}

		ray r(pos, velocityDir);
		float3 reflectedVelociy = make_float3(0, 0, 0);

		bool collisionDetected = false;
		unsigned int cellId = triangleGrid.calculateCellId(pos);
		unsigned int xId = static_cast<unsigned int>(pos.x / triangleGrid.cellWidth);
		unsigned int yId = static_cast<unsigned int>(pos.y / triangleGrid.cellHeight);
		unsigned int zId = static_cast<unsigned int>(pos.z / triangleGrid.cellDepth);

		// Check all corner cases and call the appropriate function specialization
		// Ugly but fast
		if (xId < 1)
		{
			if (yId < 1)
			{
				if (zId < 1)
				{
					collisionDetected = calculateSideCollisions<0, 1, 0, 1, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else if (zId > triangleGrid.cellCountZ - 2)
				{
					collisionDetected = calculateSideCollisions<0, 1, 0, 1, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else
				{
					collisionDetected = calculateSideCollisions<0, 1, 0, 1, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
			}
			else if (yId > triangleGrid.cellCountY - 2)
			{
				if (zId < 1)
				{
					collisionDetected = calculateSideCollisions<0, 1, -1, 0, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else if (zId > triangleGrid.cellCountZ - 2)
				{
					collisionDetected = calculateSideCollisions<0, 1, -1, 0, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else
				{
					collisionDetected = calculateSideCollisions<0, 1, -1, 0, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
			}
			else
			{
				if (zId < 1)
				{
					collisionDetected = calculateSideCollisions<0, 1, -1, 1, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else if (zId > triangleGrid.cellCountZ - 2)
				{
					collisionDetected = calculateSideCollisions<0, 1, -1, 1, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else
				{
					collisionDetected = calculateSideCollisions<0, 1, -1, 1, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
			}
		}
		else if (xId > triangleGrid.cellCountX - 2)
		{
			if (yId < 1)
			{
				if (zId < 1)
				{
					collisionDetected = calculateSideCollisions<-1, 0, 0, 1, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else if (zId > triangleGrid.cellCountZ - 2)
				{
					collisionDetected = calculateSideCollisions<-1, 0, 0, 1, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else
				{
					collisionDetected = calculateSideCollisions<-1, 0, 0, 1, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
			}
			else if (yId > triangleGrid.cellCountY - 2)
			{
				if (zId < 1)
				{
					collisionDetected = calculateSideCollisions<-1, 0, -1, 0, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else if (zId > triangleGrid.cellCountZ - 2)
				{
					collisionDetected = calculateSideCollisions<-1, 0, -1, 0, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else
				{
					collisionDetected = calculateSideCollisions<-1, 0, -1, 0, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
			}
			else
			{
				if (zId < 1)
				{
					collisionDetected = calculateSideCollisions<-1, 0, -1, 1, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else if (zId > triangleGrid.cellCountZ - 2)
				{
					collisionDetected = calculateSideCollisions<-1, 0, -1, 1, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else
				{
					collisionDetected = calculateSideCollisions<-1, 0, -1, 1, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
			}
		}
		else
		{
			if (yId < 1)
			{
				if (zId < 1)
				{
					collisionDetected = calculateSideCollisions<-1, 1, 0, 1, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else if (zId > triangleGrid.cellCountZ - 2)
				{
					collisionDetected = calculateSideCollisions<-1, 1, 0, 1, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else
				{
					collisionDetected = calculateSideCollisions<-1, 1, 0, 1, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
			}
			else if (yId > triangleGrid.cellCountY - 2)
			{
				if (zId < 1)
				{
					collisionDetected = calculateSideCollisions<-1, 1, -1, 0, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else if (zId > triangleGrid.cellCountZ - 2)
				{
					collisionDetected = calculateSideCollisions<-1, 1, -1, 0, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else
				{
					collisionDetected = calculateSideCollisions<-1, 1, -1, 0, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
			}
			else
			{
				if (zId < 1)
				{
					collisionDetected = calculateSideCollisions<-1, 1, -1, 1, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else if (zId > triangleGrid.cellCountZ - 2)
				{
					collisionDetected = calculateSideCollisions<-1, 1, -1, 1, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else
				{
					collisionDetected = calculateSideCollisions<-1, 1, -1, 1, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
			}
		}

		float3 relativePosition = pos - (pos + r.t * r.direction);
		float distanceSquared = length_squared(relativePosition);

		if (collisionDetected && distanceSquared <= veinImpactDistance * veinImpactDistance)
		{
			// handle particle on collision
			if (distanceSquared > veinImpactMinimalForceDistance * veinImpactMinimalForceDistance)
			{
				physics::addResilientForceOnCollision(relativePosition, velocity, distanceSquared,
					boundingSpheresModel[bloodCellmodelStart + (particleId - particlesStart) % particlesInBloodCell], particleId, 0.5f, bloodCells.particles.forces);
			}

			float speed = length(velocity);
			velocity = velocityCollisionDamping * speed * reflectedVelociy;
			bloodCells.particles.velocities.set(particleId, velocity);

			// handle vein on collision
			float3 ds = 0.8f * velocityDir;
			unsigned int vertexIndex0 = triangles.getIndex(r.objectIndex, vertex0);
			unsigned int vertexIndex1 = triangles.getIndex(r.objectIndex, vertex1);
			unsigned int vertexIndex2 = triangles.getIndex(r.objectIndex, vertex2);

			float3 v0 = triangles.positions.get(vertexIndex0);
			float3 v1 = triangles.positions.get(vertexIndex1);
			float3 v2 = triangles.positions.get(vertexIndex2);

			float3 baricentric = calculateBaricentric(pos + r.t * r.direction, v0, v1, v2);

			// TODO:
			// Can these lines generate concurrent write conflicts? Unlikely but not impossible. Think about it. - Filip
			// Here we probably should use atomicAdd. - Hubert
			// move triangle a bit
			triangles.forces.add(vertexIndex0, baricentric.x * ds);
			triangles.forces.add(vertexIndex1, baricentric.y * ds);
			triangles.forces.add(vertexIndex2, baricentric.z * ds);
		}

	set_particle_values:

		physics::propagateForcesInParticles(particleId, bloodCells, velocity, initialVelocity);

		return;
	}
	#pragma endregion

}
