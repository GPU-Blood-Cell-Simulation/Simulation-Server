#pragma once
#include "../utilities/math.cuh"
#include "../config/simulation.hpp"
#include "../config/physics.hpp"
#include "../utilities/cuda_vec3.cuh"
#include "../objects/blood_cells.cuh"

// determine method used for differentiation
// if not defined, heun's method will be used
//#define USE_RUNGE_KUTTA_FOR_PARTICLE

#define USE_EULER_FOR_VEIN

namespace physics
{

	/// <summary>
	/// Computes force of spring mass model with damping
	/// </summary>
	/// <param name="dP"> - relative distance p1 - p2, where p1 is particle for which we calculate force</param>
	/// <param name="dv"> - relative velocity v1 - v2, where v1 is particle for which we calculate force</param>
	/// <param name="springLength">rest length of a spring</param>
	/// <returns></returns>
	__device__ inline float springMassForceWithDampingForParticle(float3 dP, float3 dv, float springLength)
	{
		return (length(dP) - springLength) * particle_k_sniff + dot(normalize(dP), dv) * particle_d_fact;
	}

	__device__ inline float springMassForceWithDampingForVein(float3 p1, float3 p2, float3 v1, float3 v2, float springLength)
	{
		return (length(p1 - p2) - springLength) * vein_k_sniff + dot(normalize(p1 - p2), (v1 - v2)) * vein_d_fact;
	}

	/// <summary>
	/// Method to calculate resilient force component from one spring in blood cell graph
	/// </summary>
	/// <param name="dP"> - relative distance p1 - p2, where p1 is particle for which we calculate force</param>
	/// <param name="dv"> - relative velocity v1 - v2, where v1 is particle for which we calculate force</param>
	/// <param name="f1"> - force acting on particle p1 before</param>
	/// <param name="springLength">rest length of a spring</param>
	/// <param name="position"> - calculated position for next frame</param>
	/// <param name="velocity"> - calculated velocity for next frame</param>
	/// <returns></returns>
	__device__ inline float3 calculateParticlesSpringForceComponent(float3 dP, float3 dv, float3 f1, float3 f2, float springLength, float3& position, float3& velocity)
	{
		float3 normalizedShift = normalize(-1*dP);
#ifdef USE_RUNGE_KUTTA_FOR_PARTICLE
		float3 k1_x = dt * dv;
		float3 k1_v = dt * f1;

		float3 k2_x = dt * (dv + k1_x / 2);
		float3 k2_v = dt * springMassForceWithDampingForParticle(dP + k1_x / 2.0f, dv + k1_v / 2.0f, springLength)* normalizedShift;

		float3 k3_x = dt * (dv + k2_x / 2);
		float3 k3_v = dt * springMassForceWithDampingForParticle(dP + k2_x / 2.0f, dv + k2_v  / 2.0f, springLength) * normalizedShift;

		float3 k4_x = dt * (dv + k3_x);
		float3 k4_v = dt * springMassForceWithDampingForParticle(dP + k3_x, dv + k3_v, springLength) * normalizedShift;

		position = (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6.0f;
		velocity = (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0f;

		//return springMassForceWithDampingForParticle(dP, dv, springLength) * normalizedShift;
		return 2 * k2_v + 2 * k3_v + k4_v;
#else
		position = velocity = { 0,0,0 };
		return springMassForceWithDampingForParticle(dP, dv + dt*(f1 - f2), springLength) * normalizedShift;
#endif
	}

	__device__ inline float calculateVeinSpringForceComponent(float3 p1, float3 p2, float3 v1, float3 v2, float springLength)
	{
#ifdef USE_EULER_FOR_VEIN
		return springMassForceWithDampingForVein(p1, p2, v1, v2, springLength);
#else
#endif
	}

	/// <summary>
	/// gathers all other forces that influence on particle
	/// </summary>
	/// <param name="v">particle velocity</param>
	/// <returns></returns>
	__device__ inline float3 accumulateEnvironmentForcesForParticles(float3 v)
	{
		return make_float3(Gx, Gy, Gz) - viscous_damping * v;
	}


	__device__ inline void addResilientForceOnCollision(float3 relativePosition, float3 relativeVelocity, float distanceSquared, float radius, unsigned int particleId, float intensityCoefficient, cudaVec3& forces)
	{
		float3 relativeDirection = normalize(relativePosition);

		float3 tangentialVelocity = relativeVelocity - dot(relativeVelocity, relativeDirection) * relativeDirection;

		float3 springForce = -collisionSpringCoeff * (radius * 2 - sqrtf(distanceSquared)) * relativeDirection;
		float3 damplingForce = collisionDampingCoeff * relativeVelocity;
		float3 shearForce = collistionShearCoeff * tangentialVelocity;

		// Uncoalesced writes - area for optimization
		forces.add(particleId, intensityCoefficient * (springForce + damplingForce + shearForce));
	}

	__device__ inline void propagateForcesInParticles(unsigned int particleId, BloodCells& bloodCells, float3 velocity, float3 initialVelocity)
	{

		float3 F = bloodCells.particles.forces.get(particleId);
		// propagate particle forces into velocities
		velocity = velocity + dt * F;
		bloodCells.particles.velocities.set(particleId, velocity);

		// propagate velocities into positions
#ifdef USE_RUNGE_KUTTA_FOR_PARTICLE
		float3 k1_x = dt * velocity;
		float3 k2_x = dt * (velocity + k1_x / 2);
		float3 k3_x = dt * (velocity + k2_x / 2);
		float3 k4_x = dt * (velocity + k3_x);
		bloodCells.particles.positions.add(particleId, (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6.0f);
#else
		// using Heun's method
		bloodCells.particles.positions.add(particleId, 0.5f * dt * (velocity + initialVelocity));
#endif
	}
}