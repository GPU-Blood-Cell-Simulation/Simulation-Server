#ifndef BLOOD_CELLS_H
#define BLOOD_CELLS_H

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
	cudaVec3 particleCenters;
	float* initialRadiuses;
	float* dev_springGraph;

	BloodCells();
	BloodCells(const BloodCells& other);
	~BloodCells();

	/// <summary>
	/// Perform gathering forces contribution from neighbour particles for every particle
	/// </summary>
	void gatherForcesFromNeighbors(const std::array<cudaStream_t, bloodCellTypeCount>& streams);
};

#endif