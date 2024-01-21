#include "vein_triangles.cuh"

#include "../simulation/physics.cuh"
#include "../config/vein_definition.hpp"
#include "../utilities/cuda_handle_error.cuh"
#include "../utilities/math.cuh"

#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void calculateCentersKernel(cudaVec3 positions, unsigned int* indices, cudaVec3 centers, int triangleCount)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= triangleCount)
		return;
	float3 vv1 = positions.get(indices[3 * id]);
	float3 vv2 = positions.get(indices[3 * id + 1]);
	float3 vv3 = positions.get(indices[3 * id + 2]);

	float x = (vv1.x + vv2.x + vv3.x) / 3;
	float y = (vv1.y + vv2.y + vv3.y) / 3;
	float z = (vv1.z + vv2.z + vv3.z) / 3;
	centers.set(id, make_float3(x, y, z));
}

void VeinTriangles::calculateCenters(int blocks, int threadsPerBlock)
{
	calculateCentersKernel << <blocks, threadsPerBlock >> > (positions[0], indices[0], centers, triangleCount);
}

VeinTriangles::VeinTriangles()
{
	std::vector<float> vx(vertexCount);
	std::vector<float> vy(vertexCount);
	std::vector<float> vz(vertexCount);

	int iter = 0;
	std::for_each(veinPositions.begin(), veinPositions.end(), [&](auto& v)
		{
			vx[iter] = v.x;
			vy[iter] = v.y;
			vz[iter++] = v.z;
		});

	
	for (int i = 0; i < gpuCount; i++)
	{
		cudaSetDevice(i);
		// allocate
		CUDACHECK(cudaMalloc((void**)&(indices[i]), 3 * triangleCount * sizeof(int)));
		// copy
		CUDACHECK(cudaMemcpy(positions[i].x, vx.data(), vertexCount * sizeof(float), cudaMemcpyHostToDevice));
		CUDACHECK(cudaMemcpy(positions[i].y, vy.data(), vertexCount * sizeof(float), cudaMemcpyHostToDevice));
		CUDACHECK(cudaMemcpy(positions[i].z, vz.data(), vertexCount * sizeof(float), cudaMemcpyHostToDevice));
		CUDACHECK(cudaMemcpy(indices[i], veinIndices.data(), veinIndexCount * sizeof(int), cudaMemcpyHostToDevice));
	}
	cudaSetDevice(0);

	// centers
	int threadsPerBlock = triangleCount > 1024 ? 1024 : triangleCount;
	int blocks = std::ceil(static_cast<float>(triangleCount) / threadsPerBlock);
	calculateCenters(triangleCount > 1024 ? 1024 : triangleCount, std::ceil(static_cast<float>(triangleCount) / threadsPerBlock));
}

VeinTriangles::VeinTriangles(const VeinTriangles& other) : isCopy(true),
positions(other.positions), velocities(other.velocities), forces(other.forces), indices(other.indices), centers(other.centers), neighbors(other.neighbors)
{
	//std::cout << "triangles copy\n";
}

VeinTriangles::~VeinTriangles()
{
	if (!isCopy)
	{
		for (int i = 0; i < gpuCount; i++)
		{
			CUDACHECK(cudaFree(indices[i]));
		}
	}
	CUDACHECK(cudaSetDevice(0));
}

__global__ void propagateForcesIntoPositionsKernel(VeinTriangles triangles)
{
	int vertex = blockDim.x * blockIdx.x + threadIdx.x;

	if (vertex >= triangles.vertexCount)
		return;

	// propagate forces into velocities
	triangles.velocities[0].add(vertex, dt * triangles.forces[0].get(vertex));

	// propagate velocities into positions
	triangles.positions[0].add(vertex, dt * triangles.velocities[0].get(vertex));
}

/// <summary>
/// Propagate forces -> velocities and velocities->positions
/// </summary>
void VeinTriangles::propagateForcesIntoPositions(int blocks, int threadsPerBlock)
{
	propagateForcesIntoPositionsKernel << <blocks, threadsPerBlock >> > (*this);

	for (int i = 0; i < gpuCount; i++)
	{
		CUDACHECK(cudaSetDevice(i));
		CUDACHECK(cudaMemset(forces[i].x, 0, vertexCount * sizeof(float)));
		CUDACHECK(cudaMemset(forces[i].y, 0, vertexCount * sizeof(float)));
		CUDACHECK(cudaMemset(forces[i].z, 0, vertexCount * sizeof(float)));
	}
	CUDACHECK(cudaSetDevice(0));
}


/// <summary>
/// Update the tempForceBuffer based on forces applied onto 4 neighboring vertices in 2D space uisng elastic springs
/// </summary>
/// <param name="force">Vertex force vector</param>horizontalLayers
/// <param name="tempForceBuffer">Temporary buffer necessary to synchronize</param>
/// <returns></returns>
__global__ static void gatherForcesKernel(int gpuId, VeinTriangles triangles)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= triangles.vertexCount)
		return;

	float springLength = 0;
	float springForce = 0;
	float3 neighborPosition;

	float3 vertexPosition = triangles.positions[gpuId].get(id);
	float3 vertexVelocity = triangles.velocities[gpuId].get(id);
	float3 vertexForce = { 0,0,0 };
	//printf("test: %i \n", triangles.neighbors[gpuId].data[0].ids[id]);

	// For each possible neighbor check if we are attached by a spring
	#pragma unroll
	for (auto&& [neighborIds, springLengths] : triangles.neighbors[gpuId].data)
	{
		int neighborId = neighborIds[id];
		if (neighborId != -1)
		{
		 	springLength = springLengths[id];
			//printf("neighborId: %f\n", neighborId);		
		 	neighborPosition = triangles.positions[gpuId].get(neighborId);
		 	springForce = physics::springMassForceWithDampingForVein(vertexPosition, neighborPosition, vertexVelocity, triangles.velocities[gpuId].get(neighborId), springLength);
		 	vertexForce = vertexForce + springForce * normalize(neighborPosition - vertexPosition);
		}
	}	
	triangles.forces[gpuId].add(id, vertexForce);
}

/// <summary>
/// Gather forces from neighboring vertices, synchronize and then update forces for each vertex
/// </summary>
void VeinTriangles::gatherForcesFromNeighbors(int gpuId, int blocks, int threadsPerBlock)
{
	// ERROR HERE
	//gatherForcesKernel << <blocks, threadsPerBlock >> > (gpuId, *this);
	CUDACHECK(cudaDeviceSynchronize());
}

#ifdef MULTI_GPU

using namespace nccl;

void VeinTriangles::broadcastPositionsAndVelocities(ncclComm_t* comms, cudaStream_t* streams)
{
	// Broadcast positions
	NCCLCHECK(ncclGroupStart());
	broadcast(positions, vertexCount, ncclFloat, comms, streams);
	broadcast(velocities, vertexCount, ncclFloat, comms, streams);
	NCCLCHECK(ncclGroupEnd());
	// Manually zeroing forces on each gpu is cheaper than broadcasting them
	for (int i = 0; i < gpuCount; i++)
	{
		CUDACHECK(cudaSetDevice(i));
		CUDACHECK(cudaMemset(forces[i].x, 0, vertexCount * sizeof(float)));
		CUDACHECK(cudaMemset(forces[i].y, 0, vertexCount * sizeof(float)));
		CUDACHECK(cudaMemset(forces[i].z, 0, vertexCount * sizeof(float)));

		CUDACHECK(cudaStreamSynchronize(streams[i]));
	}
	CUDACHECK(cudaSetDevice(0));
}

void VeinTriangles::reduceForces(ncclComm_t* comms, cudaStream_t* streams)
{
	NCCLCHECK(ncclGroupStart());
	reduce(forces, vertexCount, ncclFloat, comms, streams);
	NCCLCHECK(ncclGroupEnd());
	sync(streams);
}
#endif
