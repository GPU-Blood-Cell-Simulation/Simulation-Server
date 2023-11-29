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
			float3 normal = normalize(cross(edge2, edge1));
			reflectionVector = r.direction - 2 * dot(r.direction, normal) * normal;
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


	
	// 1. Calculate collisions between particles and vein triangles
	// 2. Propagate forces into velocities and velocities into positions. Reset forces to 0 afterwards
	template<>
	__global__ void detectVeinCollisionsAndPropagateParticles<UniformGrid>(BloodCells bloodCells, VeinTriangles triangles, UniformGrid triangleGrid)
	{
		int particleId = blockDim.x * blockIdx.x + threadIdx.x;

		if (particleId >= particleCount)
			return;

		float3 F = bloodCells.particles.forces.get(particleId);
		float3 velocity = bloodCells.particles.velocities.get(particleId);
		float3 pos = bloodCells.particles.positions.get(particleId);

		// TEST
		//velocity = velocity + float3{ 0, 0.1f , 0 };
		//return;
		

		// propagate particle forces into velocities
		velocity = velocity + dt * F;
		
		// TODO: is there a faster way to calculate this?
		/*if (velocity.x != 0 && velocity.y != 0 && velocity.z != 0)
			goto set_particle_values;*/

		float3 velocityDir = normalize(velocity);

		ray r(pos, velocityDir);
		float3 reflectedVelociy = make_float3(0, 0, 0);

		// TODO: fix this
		bool collisionDetected = false;
		unsigned int cellId = triangleGrid.calculateCellId(pos);
		unsigned int xId = static_cast<unsigned int>(pos.x / triangleGrid.cellWidth);
		unsigned int yId = static_cast<unsigned int>(pos.y / triangleGrid.cellHeight);
		unsigned int zId = static_cast<unsigned int>(pos.z / triangleGrid.cellDepth);

		// cubical bounds
		if (modifyVelocityIfPositionOutOfBounds(pos, velocity, velocityDir)) {
			goto set_particle_values;
		}

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

		if (collisionDetected && length(pos - (pos + r.t * r.direction)) <= 5.0f)
		{
			float3 ds = 0.8f * velocityDir;
			float speed = length(velocity);
			velocity = velocityCollisionDamping * speed * reflectedVelociy;

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

		bloodCells.particles.velocities.set(particleId, velocity);

		// propagate velocities into positions
		bloodCells.particles.positions.add(particleId, dt * velocity);
		
		// zero forces
		bloodCells.particles.forces.set(particleId, make_float3(0, 0, 0));
	}

	// 1. Calculate collisions between particles and vein triangles
	// 2. Propagate forces into velocities and velocities into positions. Reset forces to 0 afterwards
	template<>
	__global__ void detectVeinCollisionsAndPropagateParticles<NoGrid>(BloodCells bloodCells, VeinTriangles triangles, NoGrid triangleGrid)
	{
		int particleId = blockDim.x * blockIdx.x + threadIdx.x;

		if (particleId >= particleCount)
			return;

		// propagate force into velocities
		float3 F = bloodCells.particles.forces.get(particleId);
		float3 velocity = bloodCells.particles.velocities.get(particleId);
		float3 pos = bloodCells.particles.positions.get(particleId);

		velocity = velocity + dt * F;
		float3 velocityDir = normalize(velocity);

		ray r(pos, velocityDir);
		float3 reflectedVelociy = make_float3(0, 0, 0);

		bool collisionOccured = false;

		// cubical bounds
		if (modifyVelocityIfPositionOutOfBounds(pos, velocity, velocityDir)) {
			goto set_particle_values;
		}

		
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

		if (collisionOccured && length(pos - (pos + r.t * r.direction)) <= 5.0f)
		{
			// triangles move vector, 2 is experimentall constant
			float3 ds = 0.8f * velocityDir;

			float speed = length(velocity);
			velocity = velocityCollisionDamping * speed * reflectedVelociy;

			unsigned int vertexIndex0 = triangles.getIndex(r.objectIndex, vertex0);
			unsigned int vertexIndex1 = triangles.getIndex(r.objectIndex, vertex1);
			unsigned int vertexIndex2 = triangles.getIndex(r.objectIndex, vertex2);

			float3 v0 = triangles.positions.get(vertexIndex0);
			float3 v1 = triangles.positions.get(vertexIndex1);
			float3 v2 = triangles.positions.get(vertexIndex2);
			float3 baricentric = calculateBaricentric(pos + r.t * r.direction, v0, v1, v2);

			// move triangle a bit
			// here we probably should use atomicAdd
			triangles.positions.add(vertexIndex0, baricentric.x * ds);
			triangles.positions.add(vertexIndex1, baricentric.y * ds);
			triangles.positions.add(vertexIndex2, baricentric.z * ds);
		}

	set_particle_values:

		bloodCells.particles.velocities.set(particleId, velocity);

		// propagate velocities into positions
		bloodCells.particles.positions.add(particleId, dt * velocity);

		// zero forces
		bloodCells.particles.forces.set(particleId, make_float3(0, 0, 0));
	}


	// because floats cannot be template parameters
	// I have put fixed boundary parameters inside this function: 
	// (xMin=0, xMax=width, xMin=0.15*height, yMax=0.85*height, zMin=0, zMax=depth)
	// Keep in mind that yMin, yMax values are strictly bounded due to
	// position of our vein in cubical space (lower and upper vein bounds are at 0.1 and 0.9 of height)
	// (I took 0.05 margin to support situation of intensified falling out of bloodCells at the both ends of vein)
	// these values might have been changed in the future !
	__device__ bool modifyVelocityIfPositionOutOfBounds(float3& position, float3& velocity, float3 normalizedVelocity)
	{
		// experimental value
		// I had one situation of "Position out of bounds" log from calculateCellId function
		// when EPS was 0.001f
		constexpr float EPS = 0.01f;

		float3 newPosition = position + dt * velocity;

		if (newPosition.x < EPS) {

			float dx = EPS - newPosition.x;
			float takeBackLength = dx / normalizedVelocity.x;
			position = position - takeBackLength * normalizedVelocity;
			velocity.x *= -1;
			return true;
		}
		else if (newPosition.x > width - EPS) {

			float dx = newPosition.x - width + EPS;
			float takeBackLength = dx / normalizedVelocity.x;
			position = position - takeBackLength * normalizedVelocity;
			velocity.x *= -1;
			return true;
		}

		if (newPosition.y < 0.15f * height + EPS) {

			float dy = EPS - newPosition.y;
			float takeBackLength = dy / normalizedVelocity.y;
			position = position - takeBackLength * normalizedVelocity;
			velocity.y *= -1;
			return true;
		}
		else if (newPosition.y > 0.85f * height - EPS) {

			float dy = newPosition.y - height + EPS;
			float takeBackLength = dy / normalizedVelocity.y;
			position = position - takeBackLength * normalizedVelocity;
			velocity.y *= -1;
			return true;
		}

		if (newPosition.z < EPS) {

			float dz = EPS - newPosition.z;
			float takeBackLength = dz / normalizedVelocity.z;
			position = position - takeBackLength * normalizedVelocity;
			velocity.z *= -1;
			return true;
		}
		else if (newPosition.z > depth - EPS) {

			float dz = newPosition.z - depth + EPS;
			float takeBackLength = dz / normalizedVelocity.z;
			position = position - takeBackLength * normalizedVelocity;
			velocity.z *= -1;
			return true;
		}
		return false;
	}

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

}