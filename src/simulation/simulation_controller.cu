#include "simulation_controller.cuh"

#include "../meta_factory/blood_cell_factory.hpp"
#include "../meta_factory/vein_factory.hpp"
#include "../objects/particles.cuh"
#include "particle_collisions.cuh"
#include "../utilities/cuda_handle_error.cuh"
#include "vein_collisions.cuh"
#include "vein_end.cuh"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <limits>
#include <thread>

#include <iostream>

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
		for (int i = 0; i < gpuCount; i++)
		{
			for (int j = 0; j < bloodCellTypeCount; j++)
			{
				CUDACHECK(cudaSetDevice(i));
				CUDACHECK(cudaStreamCreate(&streams[i][j]));
			}
		}
		CUDACHECK(cudaSetDevice(0));
		// Set up random seeds
		CUDACHECK(cudaMalloc(&devStates, particleCount * sizeof(curandState)));
		srand(static_cast<unsigned int>(time(0)));
		int seed = rand();

		setupCurandStatesKernel << <bloodCellsThreads.blocks, bloodCellsThreads.threadsPerBlock >> > (devStates, seed);

		// Generate random particle positions
		for (int i = 0; i < gpuCount; i++)
		{
			cudaSetDevice(i);
			CUDACHECK(cudaMalloc((void**)&(cellModelsBoundingSpheres[i]), particleDistinctCellsCount * sizeof(float)));
		}
		cudaSetDevice(0);
		generateBoundingSpheres();
		generateRandomPositions();
		CUDACHECK(cudaDeviceSynchronize());

	}

	SimulationController::~SimulationController()
	{
		for (int i = 0; i < gpuCount; i++)
		{
			cudaSetDevice(i);
			CUDACHECK(cudaFree(cellModelsBoundingSpheres[i]));
			for (int j = 0; j < bloodCellTypeCount; j++)
			{
				CUDACHECK(cudaStreamDestroy(streams[i][j]));
			}
		}
		cudaSetDevice(0);
		CUDACHECK(cudaFree(devStates));
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
		
			CUDACHECK(cudaMemcpy(bloodCellModels.x, hostModels[0].data(), particleDistinctCellsCount * sizeof(float), cudaMemcpyHostToDevice));
			CUDACHECK(cudaMemcpy(bloodCellModels.y, hostModels[1].data(), particleDistinctCellsCount * sizeof(float), cudaMemcpyHostToDevice));
			CUDACHECK(cudaMemcpy(bloodCellModels.z, hostModels[2].data(), particleDistinctCellsCount * sizeof(float), cudaMemcpyHostToDevice));

		std::array<float, particleDistinctCellsCount> particleRadiuses;
		for (int i = 0; i < bloodCellTypeCount; ++i)
		{
			int modelStart = bloodCellModelStarts[i];
			int modelSize = i + 1 == bloodCellTypeCount ? particleDistinctCellsCount - bloodCellModelStarts[i] :
				(i == 0 ? bloodCellModelStarts[i + 1] : bloodCellModelStarts[i + 1] - bloodCellModelStarts[i]);
			
			smallestRadiusInType[i] = std::numeric_limits<float>::max();
			float3 center {0,0,0};
			for (int j = 0; j < modelSize; ++j) {
				for (int k = 0; k < modelSize; ++k) {
					if (j != k) {
						float length = sqrt(pow((hostModels[0][modelStart + j] - hostModels[0][modelStart + k]), 2) + pow((hostModels[1][modelStart + j] -
							hostModels[1][modelStart + k]), 2) + pow((hostModels[2][modelStart + j] - hostModels[2][modelStart + k]), 2)) / (2 * boundingSpheresCoeff);

						if (length < boundingSpheres[modelStart + j])
							boundingSpheres[modelStart + j] = length;
						if (length < smallestRadiusInType[i])
							smallestRadiusInType[i] = length;
					}
				}
				center = center + make_float3(hostModels[0][modelStart + j], hostModels[1][modelStart + j], hostModels[2][modelStart + j]);
			}
			center = center/modelSize;
			for(int j = 0 ; j < modelSize; ++j)
				particleRadiuses[modelStart + j] = length(make_float3(hostModels[0][modelStart + j], hostModels[1][modelStart + j], hostModels[2][modelStart + j]) - center);
		}
		for (int i = 0; i < gpuCount; i++)
		{
			cudaSetDevice(i);
			CUDACHECK(cudaMemcpy(cellModelsBoundingSpheres[i], boundingSpheres.data(), particleDistinctCellsCount * sizeof(float), cudaMemcpyHostToDevice));
			CUDACHECK(cudaMemcpy(bloodCells.initialRadiuses[i], particleRadiuses.data(), particleDistinctCellsCount * sizeof(float), cudaMemcpyHostToDevice));
		}
		cudaSetDevice(0);
	}

	// Generate initial positions and velocities of particles
	void SimulationController::generateRandomPositions()
	{
		cudaVec3 initialPositions(bloodCellCount);
		cudaVec3 initialVelocities(bloodCellCount);

		// Generate random positions and velocity vectors
		generateRandomPositonsAndVelocitiesKernel<bloodCellCount> << <  bloodCellsThreads.blocks, bloodCellsThreads.threadsPerBlock >> > (devStates, initialPositions, initialVelocities);

		std::array<float, bloodCellCount> xpos;
		std::array<float, bloodCellCount> ypos;
		std::array<float, bloodCellCount> zpos;

		CUDACHECK(cudaMemcpy(xpos.data(), initialPositions.x, bloodCellCount * sizeof(float), cudaMemcpyDeviceToHost));
		CUDACHECK(cudaMemcpy(ypos.data(), initialPositions.y, bloodCellCount * sizeof(float), cudaMemcpyDeviceToHost));
		CUDACHECK(cudaMemcpy(zpos.data(), initialPositions.z, bloodCellCount * sizeof(float), cudaMemcpyDeviceToHost));

		for (int i = 0; i < bloodCellCount; ++i)
			initialCellPositions[i] = {xpos[i], ypos[i], zpos[i]};

		using IndexList = mp_iota_c<bloodCellTypeCount>;
		mp_for_each<IndexList>([&](auto i)
			{
				using BloodCellDefinition = mp_at_c<BloodCellList, i>;
				constexpr int particlesStart = particleStarts[i];
				constexpr int bloodCellTypeStart = bloodCellTypesStarts[i];
				constexpr int bloodCellModelSizesStarts = bloodCellModelStarts[i];

				static CudaThreads threads(BloodCellDefinition::count * BloodCellDefinition::particlesInCell);

				setBloodCellsPositionsFromRandom<BloodCellDefinition::count, BloodCellDefinition::particlesInCell, particlesStart, bloodCellTypeStart, bloodCellModelSizesStarts>
					<< <threads.blocks, threads.threadsPerBlock, 0, streams[0][i] >> > (devStates, bloodCells.particles, bloodCellModels, initialPositions, initialVelocities);
			});
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

		particles.positions[0].x[id] = initialPositions.x[bloodCellTypeStart + relativeId / particlesInBloodCell] + bloodCellModelPosition.x[bloodCellmodelStart + relativeId % particlesInBloodCell];
		particles.positions[0].y[id] = initialPositions.y[bloodCellTypeStart + relativeId / particlesInBloodCell] + bloodCellModelPosition.y[bloodCellmodelStart + relativeId % particlesInBloodCell];
		particles.positions[0].z[id] = initialPositions.z[bloodCellTypeStart + relativeId / particlesInBloodCell] + bloodCellModelPosition.z[bloodCellmodelStart + relativeId % particlesInBloodCell];

		particles.velocities[0].x[id] = initialVelocities.x[bloodCellTypeStart + relativeId / particlesInBloodCell];
		particles.velocities[0].y[id] = initialVelocities.y[bloodCellTypeStart + relativeId / particlesInBloodCell];
		particles.velocities[0].z[id] = initialVelocities.z[bloodCellTypeStart + relativeId / particlesInBloodCell];

		particles.forces[0].x[id] = 0;
		particles.forces[0].y[id] = 0;
		particles.forces[0].z[id] = 0;
	}

	// Main simulation function, called every frame
	void SimulationController::calculateNextFrame()
	{
		static auto calculate = [&](int gpuId)
		{
			CUDACHECK(cudaSetDevice(gpuId));
			std::visit([&](auto&& g1, auto&& g2)
			{
				// Propagate triangle forces into neighbors
				triangles.gatherForcesFromNeighbors(gpuId, verticesGpuStarts[gpuId], verticesGpuEnds[gpuId],
				 	veinTrianglesThreads.blocks, veinTrianglesThreads.threadsPerBlock);
				CUDACHECK(cudaDeviceSynchronize());

				// Propagate particle forces into neighbors
				bloodCells.gatherForcesFromNeighbors(gpuId, bloodCellGpuStarts[gpuId], bloodCellGpuEnds[gpuId],
					particleGpuStarts[gpuId], particleGpuEnds[gpuId], streams[gpuId]);
				CUDACHECK(cudaDeviceSynchronize());

				// Detect particle collisions
				using IndexList = mp_iota_c<bloodCellTypeCount>;
				mp_for_each<IndexList>([&](auto i)
					{
						using BloodCellDefinition = mp_at_c<BloodCellList, i>;
						constexpr int particlesStart = particleStarts[i];
						constexpr int bloodCellModelSizesStarts = bloodCellModelStarts[i];

						CudaThreads threads(BloodCellDefinition::count * BloodCellDefinition::particlesInCell);
						calculateParticleCollisions<< < threads.blocks, threads.threadsPerBlock, 0, streams[gpuId][i] >> > 
							(gpuId, particleGpuStarts[gpuId], particleGpuEnds[gpuId],
							bloodCells, *g1, cellModelsBoundingSpheres[gpuId], BloodCellDefinition::particlesInCell, bloodCellModelSizesStarts, particlesStart);
					});
				CUDACHECK(cudaDeviceSynchronize());

				// Detect vein collisions
				using IndexList = mp_iota_c<bloodCellTypeCount>;
				mp_for_each<IndexList>([&](auto i)
					{
						using BloodCellDefinition = mp_at_c<BloodCellList, i>;
						constexpr int particlesStart = particleStarts[i];
						constexpr int bloodCellModelSizesStarts = bloodCellModelStarts[i];

						CudaThreads threads(BloodCellDefinition::count * BloodCellDefinition::particlesInCell);
						// if (gpuId > 0)
						// 	std::cout << "launch kernel\n";
						detectVeinCollisions<< <  threads.blocks, threads.threadsPerBlock, 0, streams[gpuId][i] >> > 
							(gpuId, particleGpuStarts[gpuId], particleGpuEnds[gpuId], 
							bloodCells, triangles, *g2, cellModelsBoundingSpheres[gpuId], BloodCellDefinition::particlesInCell, bloodCellModelSizesStarts, particlesStart);
					});
				CUDACHECK(cudaDeviceSynchronize());
				
			}, particleGrid, triangleGrid);
		};

#ifdef  MULTI_GPU
		std::thread threads[gpuCount];
		for (int i = 0; i < gpuCount; i++)
		{
			threads[i] = std::thread(calculate, i);
		}
		for (int i = 0; i < gpuCount; i++)
		{
			threads[i].join();
		}
		CUDACHECK(cudaSetDevice(0));
#else
		calculate(0);
#endif

	}

	void SimulationController::propagateAll()
	{
		cudaSetDevice(0);

		// Propagate forces -> velocities, velocities -> positions for particles
		bloodCells.propagateForcesIntoPositions(bloodCellsThreads.blocks, bloodCellsThreads.threadsPerBlock);
		CUDACHECK(cudaDeviceSynchronize());

		// Propagate forces -> velocities, velocities -> positions for vein triangles
		triangles.propagateForcesIntoPositions(veinVerticesThreads.blocks, veinVerticesThreads.threadsPerBlock);
		CUDACHECK(cudaDeviceSynchronize());

		if constexpr (useBloodFlow)
		{
			HandleVeinEnd(bloodCells, devStates, streams[0], bloodCellModels);
			CUDACHECK(cudaPeekAtLastError());
		}

		cudaDeviceSynchronize();
	}
}