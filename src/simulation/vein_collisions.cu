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
	__global__ void detectVeinCollisionsAndPropagateForces<UniformGrid>(BloodCells bloodCells, VeinTriangles triangles, UniformGrid triangleGrid, float* boundingSpheresModel,
		int particlesInBloodCell, int bloodCellmodelStart, int particlesStart)
	{
		int particleId = particlesStart + blockDim.x * blockIdx.x + threadIdx.x;

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

		ray r(pos, velocityDir);
		float3 reflectedVelociy = make_float3(0, 0, 0);

		modifyVelocityIfPositionOutOfBounds(pos,  pos + 0.5f * dt * (velocity + initialVelocity), velocity, velocityDir);
		physics::propagateForcesInParticles(particleId, bloodCells, velocity, initialVelocity);
		
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
					if constexpr (enableReactionForce)
					{
						float3 F = bloodCells.particles.forces.get(particleId);
						float3 responseForce = -1.0f*dot(F, r.normal)*r.normal/dot(r.normal, r.normal);
						bloodCells.particles.forces.add(particleId, responseForce);
					}
				}

				float speed = length(velocity);
				velocity = velocity_collision_damping * speed * reflectedVelociy;
				bloodCells.particles.velocities.set(particleId, velocity);

				// handle vein on collision
				float3 ds = vein_collision_force_intensity * velocityDir;
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
		}
		return;
	}

	template<>
	 __global__ void detectVeinCollisionsAndPropagateForces<NoGrid>(BloodCells bloodCells, VeinTriangles triangles, NoGrid triangleGrid, float* boundingSpheresModel,
		 int particlesInBloodCell, int bloodCellmodelStart, int particlesStart)
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
		
		modifyVelocityIfPositionOutOfBounds(pos, pos + 0.5f * dt * (velocity + initialVelocity), velocity, velocityDir);
	 	physics::propagateForcesInParticles(particleId, bloodCells, velocity, initialVelocity);

	 	{
	 		ray r(pos, velocityDir);
	 		float3 reflectedVelociy = make_float3(0, 0, 0);

	 		bool collisionOccured = false;
	 		for (int triangleId = 0; triangleId < triangleCount; ++triangleId)
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
	 				if constexpr (enableReactionForce)
					{
						float3 F = bloodCells.particles.forces.get(particleId);
						float3 responseForce = -1.0f*dot(F, r.normal)*r.normal/dot(r.normal, r.normal);
						bloodCells.particles.forces.add(particleId, responseForce);
					}
				}
	 			float speed = length(velocity);
	 			velocity = velocity_collision_damping * speed * reflectedVelociy;
	 			bloodCells.particles.velocities.set(particleId, velocity);

	 			// handle vein on collision
	 			float3 ds = 0.1f * velocityDir;
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
	 	}
	 }

	// because floats cannot be template parameters
	// I have put fixed boundary parameters inside this function: 
	// (xMin=0, xMax=width, xMin=0.15*height, yMax=0.85*height, zMin=0, zMax=depth)
	// Keep in mind that yMin, yMax values are strictly bounded due to
	// position of our vein in cubical space (lower and upper vein bounds are at 0.1 and 0.9 of height)
	// (I took 0.05 margin to support situation of intensified falling out of bloodCells at the both ends of vein)
	// these values might have been changed in the future !
	__device__ bool modifyVelocityIfPositionOutOfBounds(float3& position, float3 newPosition, float3& velocity, float3 normalizedVelocity)
	{
		// experimental value
		// I had one situation of "Position out of bounds" log from calculateCellId function
		// when EPS was 0.001f
		constexpr float EPS = 0.01f;

		//float3 newPosition = position + dt * velocity;

		if (newPosition.x < minX + EPS) {

			float dx = minX + EPS - newPosition.x;
			float takeBackLength = dx / normalizedVelocity.x;
			position = position - takeBackLength * normalizedVelocity;
			velocity.x *= -1*vein_boundaries_velocity_damping;
			return true;
		}
		else if (newPosition.x > maxX - EPS) {

			float dx = newPosition.x - maxX + EPS;
			float takeBackLength = dx / normalizedVelocity.x;
			position = position - takeBackLength * normalizedVelocity;
			velocity.x *= -1*vein_boundaries_velocity_damping;
			return true;
		}

		if (newPosition.y < minY + EPS) {

			float dy = minY + EPS - newPosition.y;
			float takeBackLength = dy / normalizedVelocity.y;
			position = position - takeBackLength * normalizedVelocity;
			velocity.y *= -1*vein_boundaries_velocity_damping;
			return true;
		}
		else if (newPosition.y > maxY - EPS) {

			float dy = newPosition.y - maxY + EPS;
			float takeBackLength = dy / normalizedVelocity.y;
			position = position - takeBackLength * normalizedVelocity;
			velocity.y *= -1*vein_boundaries_velocity_damping;
			return true;
		}

		if (newPosition.z < minZ + EPS) {

			float dz = minZ + EPS - newPosition.z;
			float takeBackLength = dz / normalizedVelocity.z;
			position = position - takeBackLength * normalizedVelocity;
			velocity.z *= -1*vein_boundaries_velocity_damping;
			return true;
		}
		else if (newPosition.z > maxZ - EPS) {

			float dz = newPosition.z - maxZ + EPS;
			float takeBackLength = dz / normalizedVelocity.z;
			position = position - takeBackLength * normalizedVelocity;
			velocity.z *= -1*vein_boundaries_velocity_damping;
			return true;
		}
		return false;
	}
}