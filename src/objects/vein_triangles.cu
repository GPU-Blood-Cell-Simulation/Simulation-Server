#include "vein_triangles.cuh"

#include "../config/simulation.hpp"
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
	calculateCentersKernel << <blocks, threadsPerBlock >> > (positions, indices, centers, triangleCount);
}

VeinTriangles::VeinTriangles()
{
	// allocate
	HANDLE_ERROR(cudaMalloc((void**)&indices, 3 * triangleCount * sizeof(int)));

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

	// copy
	HANDLE_ERROR(cudaMemcpy(indices, veinIndices.data(), veinIndexCount* sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(positions.x, vx.data(), vertexCount * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(positions.y, vy.data(), vertexCount * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(positions.z, vz.data(), vertexCount * sizeof(float), cudaMemcpyHostToDevice));

	// centers
	int threadsPerBlock = triangleCount > 1024 ? 1024 : triangleCount;
	int blocks = std::ceil(static_cast<float>(triangleCount) / threadsPerBlock);
	calculateCenters(triangleCount > 1024 ? 1024 : triangleCount, std::ceil(static_cast<float>(triangleCount) / threadsPerBlock));
}

VeinTriangles::VeinTriangles(const VeinTriangles& other) : isCopy(true), vertexCount(other.vertexCount),
positions(other.positions), velocities(other.velocities), forces(other.forces), indices(other.indices), centers(other.centers), neighbors(other.neighbors)
{}

VeinTriangles::~VeinTriangles()
{
	if (!isCopy)
	{
		HANDLE_ERROR(cudaFree(indices));
	}
}

__global__ void propagateForcesIntoPositionsKernel(VeinTriangles triangles)
{
	int vertex = blockDim.x * blockIdx.x + threadIdx.x;

	if (vertex >= triangles.vertexCount)
		return;

	// propagate forces into velocities
	triangles.velocities.add(vertex, dt * triangles.forces.get(vertex));

	// propagate velocities into positions
	triangles.positions.add(vertex, dt * triangles.velocities.get(vertex));

	// zero forces
	triangles.forces.set(vertex, make_float3(0, 0, 0));
}

/// <summary>
/// Propagate forces -> velocities and velocities->positions
/// </summary>
void VeinTriangles::propagateForcesIntoPositions(int blocks, int threadsPerBlock)
{
	propagateForcesIntoPositionsKernel << <blocks, threadsPerBlock >> > (*this);
}


/// <summary>
/// Update the tempForceBuffer based on forces applied onto 4 neighboring vertices in 2D space uisng elastic springs
/// </summary>
/// <param name="force">Vertex force vector</param>horizontalLayers
/// <param name="tempForceBuffer">Temporary buffer necessary to synchronize</param>
/// <returns></returns>
__global__ static void gatherForcesKernel(VeinTriangles triangles)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= triangles.vertexCount)
		return;

	float springLength;
	float springForce;
	float3 neighborPosition;

	float3 vertexPosition = triangles.positions.get(id);
	float3 vertexVelocity = triangles.velocities.get(id);
	float3 vertexForce = { 0,0,0 };

	// For each possible neighbor check if we are attached by a spring
	#pragma unroll
	for (auto& [neighborIds, springLengths] : triangles.neighbors.data)
	{
		int neighborId = neighborIds[id];
		if (neighborId != -1)
		{
			springLength = springLengths[id];			
			neighborPosition = triangles.positions.get(neighborId);
			springForce = triangles.calculateVeinSpringForce(vertexPosition, neighborPosition, vertexVelocity, triangles.velocities.get(neighborId), springLength);
			vertexForce = vertexForce + springForce * normalize(neighborPosition - vertexPosition);	
		}
	}	
	triangles.forces.add(id, vertexForce);
}

/// <summary>
/// Gather forces from neighboring vertices, synchronize and then update forces for each vertex
/// </summary>
void VeinTriangles::gatherForcesFromNeighbors(int blocks, int threadsPerBlock)
{
	gatherForcesKernel << <blocks, threadsPerBlock >> > (*this);
}
