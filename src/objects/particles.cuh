#pragma once

#include "../meta_factory/blood_cell_factory.hpp"
#include "../utilities/cuda_vec3.cuh"

/// <summary>
/// A structure containing the position, velocity and force vectors of all particles
/// </summary>
struct Particles
{
	cudaVec3 positions;
	cudaVec3 velocities;
	cudaVec3 forces;

	Particles() : positions(particleCount), velocities(particleCount), forces(particleCount) {}
};