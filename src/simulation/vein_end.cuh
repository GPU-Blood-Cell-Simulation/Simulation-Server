#pragma once

#include "../objects/blood_cells.cuh"
#include "../utilities/cuda_threads.hpp"

#include <curand.h>
#include <curand_kernel.h>

/// <summary>
/// Launch actions to check and react if particle is on the vein end
/// </summary>
/// <param name="cells">device blood cell data</param>
/// <param name="devStates">random states for device</param>
/// <param name="streams">cuda streams</param>
/// <param name="bloodCellModels">data for blood cell positions in different types</param>
void HandleVeinEnd(BloodCells& cells, curandState* devStates, const std::array<cudaStream_t, bloodCellTypeCount>& streams, cudaVec3& bloodCellModels);
