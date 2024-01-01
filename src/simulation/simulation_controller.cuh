#ifndef SIMULATION_H
#define SIMULATION_H

#include "../meta_factory/blood_cell_factory.hpp"
#include "../grids/grid_definition.hpp"
#include "../objects/blood_cells.cuh"
#include "../objects/vein_triangles.cuh"
#include "../utilities/cuda_threads.hpp"

#include <curand.h>
#include <curand_kernel.h>
#include <variant>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace sim
{
	class SimulationController
	{
	public:
		SimulationController(BloodCells& bloodCells, VeinTriangles& triangles, Grid particleGrid, Grid triangleGrid);
		~SimulationController();

		void calculateNextFrame();
		std::vector<glm::vec3> initialCellPositions;
		float* cellModelsBoundingSpheres;

	private:
		BloodCells& bloodCells;
		VeinTriangles& triangles;
		Grid particleGrid;
		Grid triangleGrid;

		CudaThreads bloodCellsThreads;
		CudaThreads veinVerticesThreads;
		CudaThreads veinTrianglesThreads;

		std::array<cudaStream_t, bloodCellTypeCount> streams;
		curandState* devStates = 0;
		cudaVec3 bloodCellModels{particleDistinctCellsCount};

		void generateBoundingSpheres();
		void generateRandomPositions();
	};
}

#endif