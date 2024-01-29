#pragma once

#include "../meta_factory/vein_factory.hpp"


/// <summary>
/// Represents multiple pairs of arrays on gpu. Uses RAII to handle memory management
/// </summary>
class VeinNeighbors
{
	bool isCopy = true;
	int gpuId = 0;

public:

	struct pair
	{
		int* ids = 0;
		float* springs = 0;
	};

	pair data[veinVertexMaxNeighbors];

	// allocated on host
	VeinNeighbors() = default;
	VeinNeighbors(int gpuId);

	VeinNeighbors(const VeinNeighbors& other);
	VeinNeighbors& operator=(const VeinNeighbors& other);

	~VeinNeighbors();
};