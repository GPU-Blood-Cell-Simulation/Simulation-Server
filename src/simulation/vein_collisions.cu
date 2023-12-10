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
}