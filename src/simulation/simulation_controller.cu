#include "simulation_controller.cuh"

#include "../meta_factory/blood_cell_factory.hpp"
#include "../meta_factory/vein_factory.hpp"
#include "../objects/particles.cuh"
#include "particle_collisions.cuh"
#include "../utilities/cuda_handle_error.cuh"
#include "vein_collisions.cuh"
#include "vein_end.cuh"
#include <limits>
#include <cmath>
#include <ctime>
#include <algorithm>

#define INITIAL_VELOCITY_RANDOM

namespace sim
{
	/// <summary>
	/// Setups random device states
	/// </summary>
	/// <param name="states">states pointer</param>
	/// <param name="seed">seed for random</param>
	/// <returns></returns>
	__global__ void setupCurandStatesKernel(curandState* states, unsigned long seed);
	
	/// <summary>
	/// Sets initial positions for blood cells depends on model data
	/// </summary>
	template<int bloodCellCount, int particlesInBloodCell, int particlesStart, int bloodCellStart, int bloodCellmodelStar>
	__global__ void setBloodCellsPositionsFromRandom(curandState* states, Particles particles, cudaVec3 bloodCellModelPosition, cudaVec3 initialPositions, cudaVec3 initialVelocities);

	/// <summary>
	/// Generate random initial positions and velocities for particles
	/// </summary>
	template<int totalBloodCellCount>
	__global__ void generateRandomPositonsAndVelocitiesKernel(curandState* states, cudaVec3 initialPositions, cudaVec3 initialVelocities);
	
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

		// Set up random seeds
		HANDLE_ERROR(cudaMalloc(&devStates, particleCount * sizeof(curandState)));
		srand(static_cast<unsigned int>(time(0)));
		int seed = rand();

		setupCurandStatesKernel << <bloodCellsThreads.blocks, bloodCellsThreads.threadsPerBlock >> > (devStates, seed);
		HANDLE_ERROR(cudaDeviceSynchronize());

		// Generate random particle positions
		HANDLE_ERROR(cudaMalloc((void**)&cellModelsBoundingSpheres, particleDistinctCellsCount * sizeof(float)));
		generateBoundingSpheres();
		generateRandomPositions();
	}

	SimulationController::~SimulationController()
	{
		for (int i = 0; i < bloodCellTypeCount; i++)
		{
			HANDLE_ERROR(cudaStreamDestroy(streams[i]));
		}
		HANDLE_ERROR(cudaFree(cellModelsBoundingSpheres));
		HANDLE_ERROR(cudaFree(devStates));
	}

	void SimulationController::generateBoundingSpheres()
	{
		std::array<std::array<float, particleDistinctCellsCount>, 3> hostModels;
		std::array<float, particleDistinctCellsCount> boundingSpheres{};
		boundingSpheres.fill(std::numeric_limits<float>::max());
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

		std::array<float, particleDistinctCellsCount> particleRadiuses;
		for (int i = 0; i < bloodCellTypeCount; ++i)
		{
			int modelStart = bloodCellModelStarts[i];
			int modelSize = i + 1 == bloodCellTypeCount ? particleDistinctCellsCount - bloodCellModelStarts[i] :
				(i == 0 ? bloodCellModelStarts[i + 1] : bloodCellModelStarts[i + 1] - bloodCellModelStarts[i]);

			float3 center {0,0,0};
			for (int j = 0; j < modelSize; ++j) {
				for (int k = 0; k < modelSize; ++k) {
					if (j != k) {
						float length = sqrt(pow((hostModels[0][modelStart + j] - hostModels[0][modelStart + k]), 2) + pow((hostModels[1][modelStart + j] -
							hostModels[1][modelStart + k]), 2) + pow((hostModels[2][modelStart + j] - hostModels[2][modelStart + k]), 2)) / (2 * boundingSpheresCoeff);

						if (length < boundingSpheres[modelStart + j])
							boundingSpheres[modelStart + j] = length;
					}
				}
				center = center + make_float3(hostModels[0][modelStart + j], hostModels[1][modelStart + j], hostModels[2][modelStart + j]);
			}
			center = center/modelSize;
			for(int j = 0 ; j < modelSize; ++j)
				particleRadiuses[modelStart + j] = length(make_float3(hostModels[0][modelStart + j], hostModels[1][modelStart + j], hostModels[2][modelStart + j]) - center);
		}
		HANDLE_ERROR(cudaMemcpy(cellModelsBoundingSpheres, boundingSpheres.data(), particleDistinctCellsCount * sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(bloodCells.initialRadiuses, particleRadiuses.data(), particleDistinctCellsCount * sizeof(float), cudaMemcpyHostToDevice));
	}

	// Generate initial positions and velocities of particles
	void SimulationController::generateRandomPositions()
	{
		std::vector<cudaVec3> models;
		cudaVec3 initialPositions(bloodCellCount);
		cudaVec3 initialVelocities(bloodCellCount);

		// Generate random positions and velocity vectors
		generateRandomPositonsAndVelocitiesKernel<bloodCellCount> << <  bloodCellsThreads.blocks, bloodCellsThreads.threadsPerBlock >> > (devStates, initialPositions, initialVelocities);
		HANDLE_ERROR(cudaDeviceSynchronize());

		// TODO: ugly code - use std::array
		std::array<float, bloodCellCount> xpos;
		std::array<float, bloodCellCount> ypos;
		std::array<float, bloodCellCount> zpos;

		HANDLE_ERROR(cudaMemcpy(xpos.data(), initialPositions.x, bloodCellCount * sizeof(float), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(ypos.data(), initialPositions.y, bloodCellCount * sizeof(float), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(zpos.data(), initialPositions.z, bloodCellCount * sizeof(float), cudaMemcpyDeviceToHost));

		for (int i = 0; i < bloodCellCount; ++i)
			initialCellPositions.push_back(glm::vec3(xpos[i], ypos[i], zpos[i]));

		using IndexList = mp_iota_c<bloodCellTypeCount>;
		mp_for_each<IndexList>([&](auto i)
			{
				using BloodCellDefinition = mp_at_c<BloodCellList, i>;
				constexpr int particlesStart = particleStarts[i];
				constexpr int bloodCellTypeStart = bloodCellTypesStarts[i];
				constexpr int bloodCellModelSizesStarts = bloodCellModelStarts[i];

				CudaThreads threads(BloodCellDefinition::count * BloodCellDefinition::particlesInCell);
				setBloodCellsPositionsFromRandom<BloodCellDefinition::count, BloodCellDefinition::particlesInCell, particlesStart, bloodCellTypeStart, bloodCellModelSizesStarts>
					<< <threads.blocks, threads.threadsPerBlock, 0, streams[i] >> > (devStates, bloodCells.particles, bloodCellModels, initialPositions, initialVelocities);
			});
		HANDLE_ERROR(cudaDeviceSynchronize());
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
	__global__ void generateRandomPositonsAndVelocitiesKernel(curandState* states, cudaVec3 initialPositions, cudaVec3 initialVelocities)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= totalBloodCellCount)
			return;
		initialPositions.x[id] = (curand_uniform(&states[id]) - 0.5f) * 1.2f * cylinderRadius;
		initialPositions.y[id] = minSpawnY - 180.0f*curand_uniform(&states[id])*min(particleCount/1000.0f, 1.0f);
		initialPositions.z[id] = (curand_uniform(&states[id]) - 0.5f) * 1.2f * cylinderRadius;

#ifdef INITIAL_VELOCITY_RANDOM
		float verticalVelocity = randomVelocityModifier * initVelocityY;
		float complementarVelocity = abs(0.5f * verticalVelocity);
		initialVelocities.x[id] = curand_uniform(&states[id]) * 2*complementarVelocity - complementarVelocity;
		initialVelocities.y[id] = verticalVelocity;
		// simple random sign xd
		initialVelocities.z[id] = (-1*id%2)*pow(pow(complementarVelocity, 2) - pow(initialVelocities.x[id], 2), 0.5f);
#else
		initialVelocities.x[id] = initVelocityX;
		initialVelocities.y[id] = initVelocityY;
		initialVelocities.z[id] = initVelocityZ;
#endif
	}

	// Generate random positions and velocities at the beginning
	template<int bloodCellCount, int particlesInBloodCell, int particlesStart, int bloodCellTypeStart, int bloodCellmodelStart>
	__global__ void setBloodCellsPositionsFromRandom(curandState* states, Particles particles, cudaVec3 bloodCellModelPosition, cudaVec3 initialPositions, cudaVec3 initialVelocities)
	{
		int relativeId = blockIdx.x * blockDim.x + threadIdx.x;
		if (relativeId >= particlesInBloodCell * bloodCellCount)
			return;
		int id = particlesStart + relativeId;

		particles.positions.x[id] = initialPositions.x[bloodCellTypeStart + relativeId / particlesInBloodCell] + bloodCellModelPosition.x[bloodCellmodelStart + relativeId % particlesInBloodCell];
		particles.positions.y[id] = initialPositions.y[bloodCellTypeStart + relativeId / particlesInBloodCell] + bloodCellModelPosition.y[bloodCellmodelStart + relativeId % particlesInBloodCell];
		particles.positions.z[id] = initialPositions.z[bloodCellTypeStart + relativeId / particlesInBloodCell] + bloodCellModelPosition.z[bloodCellmodelStart + relativeId % particlesInBloodCell];

		particles.velocities.x[id] = initialVelocities.x[bloodCellTypeStart + relativeId / particlesInBloodCell];
		particles.velocities.y[id] = initialVelocities.y[bloodCellTypeStart + relativeId / particlesInBloodCell];
		particles.velocities.z[id] = initialVelocities.z[bloodCellTypeStart + relativeId / particlesInBloodCell];

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
				g2->calculateGrid(triangles.centers.x, triangles.centers.y, triangles.centers.z, triangleCount);

				// // 2. Propagate particle forces into neighbors
				bloodCells.gatherForcesFromNeighbors(streams);
				HANDLE_ERROR(cudaPeekAtLastError());

				// 3. Detect particle collisions
				using IndexList = mp_iota_c<bloodCellTypeCount>;
				mp_for_each<IndexList>([&](auto i)
					{
						using BloodCellDefinition = mp_at_c<BloodCellList, i>;
						constexpr int particlesStart = particleStarts[i];
						constexpr int bloodCellModelSizesStarts = bloodCellModelStarts[i];

						CudaThreads threads(BloodCellDefinition::count * BloodCellDefinition::particlesInCell);
						calculateParticleCollisions<< < threads.blocks, threads.threadsPerBlock, 0, streams[i] >> > 
							(bloodCells, *g1, cellModelsBoundingSpheres, BloodCellDefinition::particlesInCell, bloodCellModelSizesStarts, particlesStart);
					});
				HANDLE_ERROR(cudaPeekAtLastError());

				// 4. Detect vein collisions and propagate forces
				using IndexList = mp_iota_c<bloodCellTypeCount>;
				mp_for_each<IndexList>([&](auto i)
					{
						using BloodCellDefinition = mp_at_c<BloodCellList, i>;
						constexpr int particlesStart = particleStarts[i];
						constexpr int bloodCellModelSizesStarts = bloodCellModelStarts[i];

						CudaThreads threads(BloodCellDefinition::count * BloodCellDefinition::particlesInCell);
						detectVeinCollisionsAndPropagateForces<< <  threads.blocks, threads.threadsPerBlock, 0, streams[i] >> > 
							(bloodCells, triangles, *g2, cellModelsBoundingSpheres, BloodCellDefinition::particlesInCell, bloodCellModelSizesStarts, particlesStart);
					});
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
					HandleVeinEnd(bloodCells, devStates, streams, bloodCellModels);
					HANDLE_ERROR(cudaPeekAtLastError());
				}
			}, particleGrid, triangleGrid);
	}
}