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

	template<int bloodCellCount, int particlesInBloodCell, int particlesStart, int bloodCellStart>
	__global__ void setBloodCellsPositionsFromRandom(Particles particles, cudaVec3 bloodCellModelPosition, cudaVec3 initialPositions);

	template<int totalBloodCellCount>
	__global__ void generateRandomPositonskernel(curandState* states, glm::vec3 cylinderBaseCenter, cudaVec3 initialPositions);

	SimulationController::SimulationController(BloodCells& bloodCells, VeinTriangles& triangles, Grid particleGrid, Grid triangleGrid) :
		bloodCells(bloodCells), triangles(triangles), particleGrid(particleGrid), triangleGrid(triangleGrid),
		bloodCellsThreads(particleCount),
		veinVerticesThreads(triangles.vertexCount),
		veinTrianglesThreads(triangles.triangleCount)
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
		HANDLE_ERROR(cudaThreadSynchronize());

		std::vector<cudaVec3> models;
		cudaVec3 initialPositions(bloodCellCount);
		generateRandomPositonskernel<bloodCellCount> << <  bloodCellsThreads.blocks, bloodCellsThreads.threadsPerBlock >> > (devStates, cylinderBaseCenter, initialPositions);
		HANDLE_ERROR(cudaThreadSynchronize());

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

		// Generate random positions and velocity vectors
		using IndexList = mp_iota_c<bloodCellTypeCount>;
		mp_for_each<IndexList>([&](auto i)
			{

				using BloodCellDefinition = mp_at_c<BloodCellList, i>;
				constexpr int modelSize = BloodCellDefinition::particlesInCell;
				cudaVec3 g_model = cudaVec3(modelSize);
				std::vector<float> xmodel;
				std::vector<float> ymodel;
				std::vector<float> zmodel;
				using verticeIndexList = mp_iota_c<modelSize>;
				using VerticeList = typename BloodCellDefinition::Vertices;

				mp_for_each<verticeIndexList>([&](auto j)
					{
						xmodel.push_back(mp_at_c<VerticeList, j>::x);
						ymodel.push_back(mp_at_c<VerticeList, j>::y);
						zmodel.push_back(mp_at_c<VerticeList, j>::z);
					});
				HANDLE_ERROR(cudaThreadSynchronize());
				HANDLE_ERROR(cudaMemcpy(g_model.x, xmodel.data(), modelSize * sizeof(float), cudaMemcpyHostToDevice));
				HANDLE_ERROR(cudaMemcpy(g_model.y, ymodel.data(), modelSize * sizeof(float), cudaMemcpyHostToDevice));
				HANDLE_ERROR(cudaMemcpy(g_model.z, zmodel.data(), modelSize * sizeof(float), cudaMemcpyHostToDevice));
				models.push_back(g_model);
			});
		mp_for_each<IndexList>([&](auto i)
			{
				using BloodCellDefinition = mp_at_c<BloodCellList, i>;
				constexpr int particlesStart = particlesStarts[i];
				constexpr int bloodCellTypeStart = bloodCellTypesStarts[i];

				CudaThreads threads(BloodCellDefinition::count * BloodCellDefinition::particlesInCell);
				setBloodCellsPositionsFromRandom<BloodCellDefinition::count, BloodCellDefinition::particlesInCell, particlesStart, bloodCellTypeStart>
					<< <threads.blocks, threads.threadsPerBlock, 0, streams[i] >> > (bloodCells.particles, models[i], initialPositions);
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


	template<int totalBloodCellCount>
	__global__ void generateRandomPositonskernel(curandState* states, glm::vec3 cylinderBaseCenter, cudaVec3 initialPositions)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= totalBloodCellCount)
			return;
		initialPositions.x[id] = cylinderBaseCenter.x - cylinderRadius * 0.5f + curand_uniform(&states[id]) * cylinderRadius;
		initialPositions.y[id] = cylinderBaseCenter.y - cylinderRadius * 0.5f + curand_uniform(&states[id]) * cylinderRadius + cylinderHeight / 2;
		initialPositions.z[id] = cylinderBaseCenter.z - cylinderRadius * 0.5f + curand_uniform(&states[id]) * cylinderRadius;

		printf("[%d] initialPos.x=%.5f, initialPos.y=%.5f, initialPos.z=%.5f\n", id, initialPositions.x[id], initialPositions.y[id], initialPositions.z[id]);
	}

	// Generate random positions and velocities at the beginning
	template<int bloodCellCount, int particlesInBloodCell, int particlesStart, int bloodCellTypeStart>
	__global__ void setBloodCellsPositionsFromRandom(Particles particles, cudaVec3 bloodCellModelPosition, cudaVec3 initialPositions)
	{
		int relativeId = blockIdx.x * blockDim.x + threadIdx.x;
		if (relativeId >= particlesInBloodCell * bloodCellCount)
			return;
		int id = particlesStart + relativeId;

		particles.positions.x[id] = initialPositions.x[bloodCellTypeStart + relativeId / particlesInBloodCell] + bloodCellModelPosition.x[id % particlesInBloodCell];
		particles.positions.y[id] = initialPositions.y[bloodCellTypeStart + relativeId / particlesInBloodCell] + bloodCellModelPosition.y[id % particlesInBloodCell];
		particles.positions.z[id] = initialPositions.z[bloodCellTypeStart + relativeId / particlesInBloodCell] + bloodCellModelPosition.z[id % particlesInBloodCell];
		printf("[%d][%d][%d][%d][%d] particle position: x = %.5f, y = %.5f, z = %.5f\n", id, relativeId, bloodCellTypeStart, particlesStart, particlesInBloodCell, particles.positions.x[id], particles.positions.y[id], particles.positions.z[id]);

		particles.velocities.x[id] = 0;
		particles.velocities.y[id] = -10;
		particles.velocities.z[id] = 0;

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
				// TODO: possible optimization - these grisds can be calculated simultaneously
				g1->calculateGrid(bloodCells.particles, particleCount);
				g2->calculateGrid(triangles.centers.x, triangles.centers.y, triangles.centers.z, triangles.triangleCount);

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

				/*if constexpr (useBloodFlow)
				{
					HandleVeinEnd(bloodCells, streams);
					HANDLE_ERROR(cudaPeekAtLastError());
				}*/

			}, particleGrid, triangleGrid);
	}
}