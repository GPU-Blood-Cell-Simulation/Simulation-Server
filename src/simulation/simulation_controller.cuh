#ifndef SIMULATION_H
#define SIMULATION_H

#include "../meta_factory/blood_cell_factory.hpp"
#include "../grids/no_grid.cuh"
#include "../grids/uniform_grid.cuh"
#include "../objects/blood_cells.cuh"
#include "../objects/vein_triangles.cuh"
#include "../utilities/cuda_threads.hpp"

#include <variant>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using Grid = std::variant<UniformGrid*, NoGrid*>;

namespace sim
{
	class SimulationController
	{
	public:
		SimulationController(BloodCells& bloodCells, VeinTriangles& triangles, Grid particleGrid, Grid triangleGrid);
		~SimulationController();

		void calculateNextFrame();
		std::vector<glm::vec3> initialCellPositions;

	private:
		BloodCells& bloodCells;
		VeinTriangles& triangles;
		Grid particleGrid;
		Grid triangleGrid;

		CudaThreads bloodCellsThreads;
		CudaThreads veinVerticesThreads;
		CudaThreads veinTrianglesThreads;

		std::array<cudaStream_t, bloodCellTypeCount> streams;

		void generateRandomPositions();
	};
}

#endif