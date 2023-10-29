#pragma once

#include "../objects/blood_cells.cuh"
#include "../utilities/cuda_threads.hpp"


void HandleVeinEnd(BloodCells& cells, const std::array<cudaStream_t, bloodCellTypeCount>& streams);
