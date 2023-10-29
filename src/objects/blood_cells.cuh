#ifndef BLOOD_CELLS_H
#define BLOOD_CELLS_H

#include "../config/physics.hpp"
#include "../meta_factory/blood_cell_factory.hpp"
#include "particles.cuh"
#include "../utilities/math.cuh"


/// <summary>
/// Contains all the particles and the springs graph
/// </summary>
class BloodCells
{
	bool isCopy = false;

public:
	Particles particles;
	float* dev_springGraph;

	BloodCells();
	BloodCells(const BloodCells& other);
	~BloodCells();

	__device__ inline float calculateParticleSpringForce(float3 p1, float3 p2, float3 v1, float3 v2, float springLength)
	{
		return (length(p1 - p2) - springLength) * particle_k_sniff + dot(normalize(p1 - p2), (v1 - v2)) * particle_d_fact;
	}

	void BloodCells::gatherForcesFromNeighbors(const std::array<cudaStream_t, bloodCellTypeCount>& streams);
};

#endif