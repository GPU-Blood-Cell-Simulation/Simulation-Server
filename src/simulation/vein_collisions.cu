#include "vein_collisions.cuh"

#include "../utilities/cuda_handle_error.cuh"
#include "../utilities/vertex_index_enum.hpp"


namespace sim
{
	__device__ ray::ray(float3 origin, float3 direction) : origin(origin), direction(direction) {}

	__device__ bool realCollisionDetection(float3 v0, float3 v1, float3 v2, ray& r, float3& reflectionVector)
	{
		constexpr float EPS = 0.000001f;
		const float3 edge1 = v1 - v0;
		const float3 edge2 = v2 - v0;

		const float3 h = cross(r.direction, edge2);
		const float a = dot(edge1, h);
		if (a > -EPS && a < EPS)
			return false; // ray parallel to triangle

		const float f = 1 / a;
		const float3 s = r.origin - v0;
		const float u = f * dot(s, h);
		if (u < 0 || u > 1)
			return false;
		const float3 q = cross(s, edge1);
		const float v = f * dot(r.direction, q);
		if (v < 0 || u + v > 1)
			return false;
		const float t = f * dot(edge2, q);
		if (t > EPS)
		{
			r.t = t;

			// this normal is oriented to the vein interior
			// it is caused by the order of vertices in triangles used to correct face culling
			// change order of edge2 and edge1 in cross product for oposite normal
			// Question: Is the situation when we should use oposite normal possible ?
			r.normal = normalize(cross(edge2, edge1));
			reflectionVector = r.direction - 2 * dot(r.direction, r.normal) * r.normal;
			return true;
		}
		return false;
	}

	__device__ float3 calculateBaricentric(float3 point, float3 v0, float3 v1, float3 v2)
	{
		float3 baricentric;
		float3 e0 = v1 - v0, e1 = v2 - v1, e2 = point - v0;
		float d00 = dot(e0, e0);
		float d01 = dot(e0, e1);
		float d11 = dot(e1, e1);
		float d20 = dot(e2, e0);
		float d21 = dot(e2, e1);
		float denom = d00 * d11 - d01 * d01;
		baricentric.x = (d11 * d20 - d01 * d21) / denom;
		baricentric.y = (d00 * d21 - d01 * d20) / denom;
		baricentric.z = 1.0f - baricentric.x - baricentric.y;
		return baricentric;
	}

	template<>
	__global__ void detectVeinCollisions<UniformGrid>(int gpuId, int gpuStart, int gpuEnd, BloodCells bloodCells, VeinTriangles triangles, UniformGrid triangleGrid, float* boundingSpheresModel,
		int bloodCellsOfType, int particlesInBloodCell, int bloodCellmodelStart, int particlesStart)
	{
		int particleId = particlesStart + blockDim.x * blockIdx.x + threadIdx.x;
		if (particleId < gpuStart || particleId >= gpuEnd || particleId >= particlesStart + bloodCellsOfType * particlesInBloodCell)
			return;

		float3 velocity = bloodCells.particles.velocities[gpuId].get(particleId);
		float3 pos = bloodCells.particles.positions[gpuId].get(particleId);
		
		float3 velocityDir = normalize(velocity);

		ray r(pos, velocityDir);
		float3 reflectedVelociy = make_float3(0, 0, 0);
		
		// TODO: fix this
		bool collisionDetected = false;
		unsigned int cellId = triangleGrid.calculateCellId(pos);

		unsigned int xId = static_cast<unsigned int>((pos.x - minX) / triangleGrid.cellWidth);
		unsigned int yId = static_cast<unsigned int>((pos.y - minY) / triangleGrid.cellHeight);
		unsigned int zId = static_cast<unsigned int>((pos.z - minZ) / triangleGrid.cellDepth);
		
		{
			// Check all corner cases and call the appropriate function specialization
			// Ugly but fast
			if (xId < 1)
			{
				if (yId < 1)
				{
					if (zId < 1)
					{
						collisionDetected = calculateSideCollisions<0, 1, 0, 1, 0, 1>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
					else if (zId > triangleGrid.cellCountZ - 2)
					{
						collisionDetected = calculateSideCollisions<0, 1, 0, 1, -1, 0>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
					else
					{
						collisionDetected = calculateSideCollisions<0, 1, 0, 1, -1, 1>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
				}
				else if (yId > triangleGrid.cellCountY - 2)
				{
					if (zId < 1)
					{
						collisionDetected = calculateSideCollisions<0, 1, -1, 0, 0, 1>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
					else if (zId > triangleGrid.cellCountZ - 2)
					{
						collisionDetected = calculateSideCollisions<0, 1, -1, 0, -1, 0>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
					else
					{
						collisionDetected = calculateSideCollisions<0, 1, -1, 0, -1, 1>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
				}
				else
				{
					if (zId < 1)
					{
						collisionDetected = calculateSideCollisions<0, 1, -1, 1, 0, 1>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
					else if (zId > triangleGrid.cellCountZ - 2)
					{
						collisionDetected = calculateSideCollisions<0, 1, -1, 1, -1, 0>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
					else
					{
						collisionDetected = calculateSideCollisions<0, 1, -1, 1, -1, 1>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
				}
			}
			else if (xId > triangleGrid.cellCountX - 2)
			{
				if (yId < 1)
				{
					if (zId < 1)
					{
						collisionDetected = calculateSideCollisions<-1, 0, 0, 1, 0, 1>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
					else if (zId > triangleGrid.cellCountZ - 2)
					{
						collisionDetected = calculateSideCollisions<-1, 0, 0, 1, -1, 0>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
					else
					{
						collisionDetected = calculateSideCollisions<-1, 0, 0, 1, -1, 1>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
				}
				else if (yId > triangleGrid.cellCountY - 2)
				{
					if (zId < 1)
					{
						collisionDetected = calculateSideCollisions<-1, 0, -1, 0, 0, 1>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
					else if (zId > triangleGrid.cellCountZ - 2)
					{
						collisionDetected = calculateSideCollisions<-1, 0, -1, 0, -1, 0>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
					else
					{
						collisionDetected = calculateSideCollisions<-1, 0, -1, 0, -1, 1>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
				}
				else
				{
					if (zId < 1)
					{
						collisionDetected = calculateSideCollisions<-1, 0, -1, 1, 0, 1>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
					else if (zId > triangleGrid.cellCountZ - 2)
					{
						collisionDetected = calculateSideCollisions<-1, 0, -1, 1, -1, 0>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
					else
					{
						collisionDetected = calculateSideCollisions<-1, 0, -1, 1, -1, 1>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
				}
			}
			else
			{
				if (yId < 1)
				{
					if (zId < 1)
					{
						collisionDetected = calculateSideCollisions<-1, 1, 0, 1, 0, 1>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
					else if (zId > triangleGrid.cellCountZ - 2)
					{
						collisionDetected = calculateSideCollisions<-1, 1, 0, 1, -1, 0>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
					else
					{
						collisionDetected = calculateSideCollisions<-1, 1, 0, 1, -1, 1>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
				}
				else if (yId > triangleGrid.cellCountY - 2)
				{
					if (zId < 1)
					{
						collisionDetected = calculateSideCollisions<-1, 1, -1, 0, 0, 1>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
					else if (zId > triangleGrid.cellCountZ - 2)
					{
						collisionDetected = calculateSideCollisions<-1, 1, -1, 0, -1, 0>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
					else
					{
						collisionDetected = calculateSideCollisions<-1, 1, -1, 0, -1, 1>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
				}
				else
				{
					if (zId < 1)
					{
						collisionDetected = calculateSideCollisions<-1, 1, -1, 1, 0, 1>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
					else if (zId > triangleGrid.cellCountZ - 2)
					{
						collisionDetected = calculateSideCollisions<-1, 1, -1, 1, -1, 0>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
					}
					else
					{
						collisionDetected = calculateSideCollisions<-1, 1, -1, 1, -1, 1>(gpuId, pos, r, reflectedVelociy, triangles, triangleGrid);
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
					// if (gpuId > 0)
					// 	printf("gpu: %d, particle: %d", gpuId, particleId);
					physics::addResilientForceOnCollision(relativePosition, velocity, distanceSquared,
						boundingSpheresModel[bloodCellmodelStart + (particleId - particlesStart) % particlesInBloodCell], particleId, 0.5f, bloodCells.particles.forces[gpuId]);
					if constexpr (enableReactionForce)
					{
						float3 F = bloodCells.particles.forces[gpuId].get(particleId);
						float3 responseForce = -1.0f*dot(F, r.normal)*r.normal/dot(r.normal, r.normal);
						bloodCells.particles.forces[gpuId].add(particleId, responseForce);
					}
				}

				float speed = length(velocity);
				// Account for division by gpuCount after ncclReduce
				float3 dv = gpuCount * (velocity_collision_damping * speed * reflectedVelociy - velocity);
				bloodCells.particles.velocities[gpuId].add(particleId, dv);

				// handle vein on collision
				float3 ds = vein_collision_force_intensity * velocityDir;
				unsigned int vertexIndex0 = triangles.getIndex(gpuId, r.objectIndex, vertex0);
				unsigned int vertexIndex1 = triangles.getIndex(gpuId, r.objectIndex, vertex1);
				unsigned int vertexIndex2 = triangles.getIndex(gpuId, r.objectIndex, vertex2);

				float3 v0 = triangles.positions[gpuId].get(vertexIndex0);
				float3 v1 = triangles.positions[gpuId].get(vertexIndex1);
				float3 v2 = triangles.positions[gpuId].get(vertexIndex2);

				float3 baricentric = calculateBaricentric(pos + r.t * r.direction, v0, v1, v2);

				// TODO:
				// Can these lines generate concurrent write conflicts? Unlikely but not impossible. Think about it. - Filip
				// Here we probably should use atomicAdd. - Hubert
				// move triangle a bit
				triangles.forces[gpuId].add(vertexIndex0, baricentric.x * ds);
				triangles.forces[gpuId].add(vertexIndex1, baricentric.y * ds);
				triangles.forces[gpuId].add(vertexIndex2, baricentric.z * ds);
			}
		}
	}

	template<>
	 __global__ void detectVeinCollisions<NoGrid>(int gpuId, int gpuStart, int gpuEnd, BloodCells bloodCells, VeinTriangles triangles, NoGrid triangleGrid, float* boundingSpheresModel,
		 int bloodCellsOfType, int particlesInBloodCell, int bloodCellmodelStart, int particlesStart)
	 {
	 	int particleId = particlesStart + blockDim.x * blockIdx.x + threadIdx.x;
		if (particleId < gpuStart || particleId >= gpuEnd || particleId >= particlesStart + bloodCellsOfType * particlesInBloodCell)
			return;

	 	// propagate force into velocities
	 	float3 F = bloodCells.particles.forces[gpuId].get(particleId);
	 	float3 velocity = bloodCells.particles.velocities[gpuId].get(particleId);
	 	float3 pos = bloodCells.particles.positions[gpuId].get(particleId);

	 	velocity = velocity + dt * F;
	 	float3 velocityDir = normalize(velocity);
		
	 	{
	 		ray r(pos, velocityDir);
	 		float3 reflectedVelociy = make_float3(0, 0, 0);

	 		bool collisionOccured = false;
	 		for (int triangleId = 0; triangleId < triangleCount; ++triangleId)
	 		{
	 			constexpr float EPS = 1e-7f;
	 			// triangle vectices and edges
	 			float3 v0 = triangles.positions[gpuId].get(triangles.getIndex(gpuId, triangleId, vertex0));
	 			float3 v1 = triangles.positions[gpuId].get(triangles.getIndex(gpuId, triangleId, vertex1));
	 			float3 v2 = triangles.positions[gpuId].get(triangles.getIndex(gpuId, triangleId, vertex2));
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
	 					boundingSpheresModel[bloodCellmodelStart + (particleId - particlesStart) % particlesInBloodCell], 0.5f, bloodCells.particles.forces[gpuId]);
	 				if constexpr (enableReactionForce)
					{
						float3 F = bloodCells.particles.forces[gpuId].get(particleId);
						float3 responseForce = -1.0f*dot(F, r.normal)*r.normal/dot(r.normal, r.normal);
						bloodCells.particles.forces[gpuId].add(particleId, responseForce);
					}
				}
	 			float speed = length(velocity);
	 			velocity = velocity_collision_damping * speed * reflectedVelociy;
	 			bloodCells.particles.velocities[gpuId].set(particleId, velocity);

	 			// handle vein on collision
	 			float3 ds = 0.1f * velocityDir;
	 			unsigned int vertexIndex0 = triangles.getIndex(gpuId, r.objectIndex, vertex0);
	 			unsigned int vertexIndex1 = triangles.getIndex(gpuId, r.objectIndex, vertex1);
	 			unsigned int vertexIndex2 = triangles.getIndex(gpuId, r.objectIndex, vertex2);

	 			float3 v0 = triangles.positions[gpuId].get(vertexIndex0);
	 			float3 v1 = triangles.positions[gpuId].get(vertexIndex1);
	 			float3 v2 = triangles.positions[gpuId].get(vertexIndex2);

	 			float3 baricentric = calculateBaricentric(pos + r.t * r.direction, v0, v1, v2);

	 			// TODO:
	 			// Can these lines generate concurrent write conflicts? Unlikely but not impossible. Think about it. - Filip
	 			// Here we probably should use atomicAdd. - Hubert
	 			// move triangle a bit
	 			triangles.forces[gpuId].add(vertexIndex0, baricentric.x * ds);
	 			triangles.forces[gpuId].add(vertexIndex1, baricentric.y * ds);
	 			triangles.forces[gpuId].add(vertexIndex2, baricentric.z * ds);
	 		}
	 	}
	 }
}