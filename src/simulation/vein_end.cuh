#pragma once

#include "../objects/blood_cells.cuh"
#include "../utilities/cuda_threads.hpp"

#include <curand.h>
#include <curand_kernel.h>


void HandleVeinEnd(BloodCells& cells, curandState* devStates, const std::array<cudaStream_t, bloodCellTypeCount>& streams, cudaVec3& bloodCellModels);
