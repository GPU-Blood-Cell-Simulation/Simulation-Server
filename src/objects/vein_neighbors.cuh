#pragma once

#include "../meta_factory/vein_factory.hpp"

/// <summary>
/// Represents multiple pairs of arrays on gpu. Uses RAII to handle memory management
/// </summary>
class VeinNeighbors
{
	bool isCopy = false;

public:

	struct pair
	{
		int* ids = 0;
		float* springs = 0;
	};

	pair data[veinVertexMaxNeighbors];

	// allocated on host
	VeinNeighbors();
	VeinNeighbors(const VeinNeighbors& other);
	~VeinNeighbors();
};