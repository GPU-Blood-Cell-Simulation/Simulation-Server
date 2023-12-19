#include "simulation_controller.cuh"

#include "../meta_factory/blood_cell_factory.hpp"
#include "../meta_factory/vein_factory.hpp"
#include "../objects/particles.cuh"
#include "particle_collisions.cuh"
#include "../utilities/cuda_handle_error.cuh"
#include "vein_collisions.cuh"
#include "vein_end.cuh"

#include <cmath>
#include <ctime>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>

namespace sim
{
	__global__ void setupCurandStatesKernel(curandState* states, unsigned long seed);

	template<int bloodCellCount, int particlesInBloodCell, int particlesStart, int bloodCellStart, int bloodCellmodelStar>
	__global__ void setBloodCellsPositionsFromRandom(Particles particles, cudaVec3 bloodCellModelPosition, cudaVec3 initialPositions);

	template<int totalBloodCellCount>
	__global__ void generateRandomPositonskernel(curandState* states, cudaVec3 initialPositions);

	SimulationController::SimulationController(BloodCells& bloodCells, VeinTriangles& triangles, Grid particleGrid, Grid triangleGrid) :
		bloodCells(bloodCells), triangles(triangles), particleGrid(particleGrid), triangleGrid(triangleGrid),
		bloodCellsThreads(particleCount),
		veinVerticesThreads(triangles.vertexCount),
		veinTrianglesThreads(triangleCount)
	{
		// Create streams
		for (int i = 0; i < bloodCellTypeCount; i++)
		{
			streams[i] = cudaStream_t();
			HANDLE_ERROR(cudaStreamCreate(&streams[i]));
		}

		// Generate random particle positions
		generateRandomPositions();
	}

	sim::SimulationController::~SimulationController()
	{
		for (int i = 0; i < bloodCellTypeCount; i++)
		{
			HANDLE_ERROR(cudaStreamDestroy(streams[i]));
		}
	}

	// Generate initial positions and velocities of particles
	void SimulationController::generateRandomPositions()
	{
		// Set up random seeds
		curandState* devStates;
		HANDLE_ERROR(cudaMalloc(&devStates, particleCount * sizeof(curandState)));
		srand(static_cast<unsigned int>(time(0)));
		int seed = rand();

		setupCurandStatesKernel << <bloodCellsThreads.blocks, bloodCellsThreads.threadsPerBlock >> > (devStates, seed);
		HANDLE_ERROR(cudaDeviceSynchronize());

		std::vector<cudaVec3> models;
		cudaVec3 initialPositions(bloodCellCount);

		// Generate random positions and velocity vectors
		generateRandomPositonskernel<bloodCellCount> << <  bloodCellsThreads.blocks, bloodCellsThreads.threadsPerBlock >> > (devStates, initialPositions);
		HANDLE_ERROR(cudaDeviceSynchronize());

		// TODO: ugly code - use std::array
		float* xpos = new float[bloodCellCount];
		float* ypos = new float[bloodCellCount];
		float* zpos = new float[bloodCellCount];

		HANDLE_ERROR(cudaMemcpy(xpos, initialPositions.x, bloodCellCount * sizeof(float), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(ypos, initialPositions.y, bloodCellCount * sizeof(float), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(zpos, initialPositions.z, bloodCellCount * sizeof(float), cudaMemcpyDeviceToHost));

		for (int i = 0; i < bloodCellCount; ++i)
			initialCellPositions.push_back(glm::vec3(xpos[i], ypos[i], zpos[i]));

		delete[] xpos;
		delete[] ypos;
		delete[] zpos;

		cudaVec3 bloodCellModels = cudaVec3(particleDistinctCellsCount);
		std::array<std::array<float, particleDistinctCellsCount>, 3> hostModels;

		using IndexList = mp_iota_c<bloodCellTypeCount>;
		mp_for_each<IndexList>([&](auto i)
			{
				using BloodCellDefinition = mp_at_c<BloodCellList, i>;
				constexpr int modelSize = BloodCellDefinition::particlesInCell;
				int modelStart = bloodCellModelStarts[i];
				using verticeIndexList = mp_iota_c<modelSize>;
				using VerticeList = typename BloodCellDefinition::Vertices;

				mp_for_each<verticeIndexList>([&](auto j)
					{
						hostModels[0][modelStart + j] = mp_at_c<VerticeList, j>::x;
						hostModels[1][modelStart + j] = mp_at_c<VerticeList, j>::y;
						hostModels[2][modelStart + j] = mp_at_c<VerticeList, j>::z;
					});
			});
		HANDLE_ERROR(cudaMemcpy(bloodCellModels.x, hostModels[0].data(), particleDistinctCellsCount * sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(bloodCellModels.y, hostModels[1].data(), particleDistinctCellsCount * sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(bloodCellModels.z, hostModels[2].data(), particleDistinctCellsCount * sizeof(float), cudaMemcpyHostToDevice));

		mp_for_each<IndexList>([&](auto i)
			{
				using BloodCellDefinition = mp_at_c<BloodCellList, i>;
				constexpr int particlesStart = particlesStarts[i];
				constexpr int bloodCellTypeStart = bloodCellTypesStarts[i];
				constexpr int bloodCellModelSizesStarts = bloodCellModelStarts[i];

				CudaThreads threads(BloodCellDefinition::count * BloodCellDefinition::particlesInCell);
				setBloodCellsPositionsFromRandom<BloodCellDefinition::count, BloodCellDefinition::particlesInCell, particlesStart, bloodCellTypeStart, bloodCellModelSizesStarts>
					<< <threads.blocks, threads.threadsPerBlock, 0, streams[i] >> > (bloodCells.particles, bloodCellModels, initialPositions);
			});
		HANDLE_ERROR(cudaDeviceSynchronize());
		HANDLE_ERROR(cudaFree(devStates));
	}

	__global__ void setupCurandStatesKernel(curandState* states, unsigned long seed)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= particleCount)
			return;
		curand_init(seed, id, 0, &states[id]);
	}

	// generate initial positions for blood cells
	template<int totalBloodCellCount>
	__global__ void generateRandomPositonskernel(curandState* states, cudaVec3 initialPositions)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= totalBloodCellCount)
			return;
		initialPositions.x[id] = (curand_uniform(&states[id]) - 0.5f) * 0.5f * cylinderRadius;
		initialPositions.y[id] = minSpawnY;
		initialPositions.z[id] = (curand_uniform(&states[id]) - 0.5f) * 0.5f * cylinderRadius;
	}

	// Generate random positions and velocities at the beginning
	template<int bloodCellCount, int particlesInBloodCell, int particlesStart, int bloodCellTypeStart, int bloodCellmodelStart>
	__global__ void setBloodCellsPositionsFromRandom(Particles particles, cudaVec3 bloodCellModelPosition, cudaVec3 initialPositions)
	{
		int relativeId = blockIdx.x * blockDim.x + threadIdx.x;
		if (relativeId >= particlesInBloodCell * bloodCellCount)
			return;
		int id = particlesStart + relativeId;

		particles.positions.x[id] = initialPositions.x[bloodCellTypeStart + relativeId / particlesInBloodCell] + bloodCellModelPosition.x[bloodCellmodelStart + relativeId % particlesInBloodCell];
		particles.positions.y[id] = initialPositions.y[bloodCellTypeStart + relativeId / particlesInBloodCell] + bloodCellModelPosition.y[bloodCellmodelStart + relativeId % particlesInBloodCell];
		particles.positions.z[id] = initialPositions.z[bloodCellTypeStart + relativeId / particlesInBloodCell] + bloodCellModelPosition.z[bloodCellmodelStart + relativeId % particlesInBloodCell];

		particles.velocities.x[id] = initVelocityX;
		particles.velocities.y[id] = initVelocityY;
		particles.velocities.z[id] = initVelocityZ;

		particles.forces.x[id] = 0;
		particles.forces.y[id] = 0;
		particles.forces.z[id] = 0;
	}

	// Main simulation function, called every frame
	void SimulationController::calculateNextFrame()
	{
		std::visit([&](auto&& g1, auto&& g2)
			{
				// 1. Calculate grids
				// TODO: possible optimization - these grids can be calculated simultaneously
				g1->calculateGrid(bloodCells.particles, particleCount);
				g2->calculateGrid(triangles.centers.x, triangles.centers.y, triangles.centers.z, triangleCount);

				// 2. Detect particle collisions
				calculateParticleCollisions << < bloodCellsThreads.blocks, bloodCellsThreads.threadsPerBlock >> > (bloodCells, *g1);
				HANDLE_ERROR(cudaPeekAtLastError());

				// 3. Propagate particle forces into neighbors

				bloodCells.gatherForcesFromNeighbors(streams);
				HANDLE_ERROR(cudaPeekAtLastError());

				// 4. Detect vein collisions and propagate forces -> velocities, velocities -> positions for particles

				detectVeinCollisionsAndPropagateParticles << < bloodCellsThreads.blocks, bloodCellsThreads.threadsPerBlock >> > (bloodCells, triangles, *g2);
				HANDLE_ERROR(cudaPeekAtLastError());

				// 5. Propagate triangle forces into neighbors
				triangles.gatherForcesFromNeighbors(veinVerticesThreads.blocks, veinVerticesThreads.threadsPerBlock);
				HANDLE_ERROR(cudaPeekAtLastError());		

				// 6. Propagate forces -> velocities, velocities -> positions for vein triangles
				triangles.propagateForcesIntoPositions(veinVerticesThreads.blocks, veinVerticesThreads.threadsPerBlock);
				HANDLE_ERROR(cudaPeekAtLastError());

				// 7. Recalculate triangles centers
				triangles.calculateCenters(veinTrianglesThreads.blocks, veinTrianglesThreads.threadsPerBlock);
				HANDLE_ERROR(cudaPeekAtLastError());

				if constexpr (useBloodFlow)
				{
					HandleVeinEnd(bloodCells, streams);
					HANDLE_ERROR(cudaPeekAtLastError());
				}

			}, particleGrid, triangleGrid);
	}
}