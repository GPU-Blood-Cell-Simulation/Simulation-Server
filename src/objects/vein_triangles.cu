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

VeinTriangles::VeinTriangles(const std::vector<glm::vec3>& vertices, const std::vector<unsigned int>& indexData, const std::tuple<float, float, float>& springLengths) :
	triangleCount(indexData.size() / 3), vertexCount(vertices.size()),
	centers(triangleCount), positions(vertexCount), velocities(vertexCount), forces(vertexCount),
	veinVertexHorizontalDistance(std::get<0>(springLengths)),
	veinVertexNonHorizontalDistances{ std::get<2>(springLengths), std::get<1>(springLengths), std::get<2>(springLengths) }
{
	// allocate
	HANDLE_ERROR(cudaMalloc((void**)&indices, 3 * triangleCount * sizeof(int)));

	std::vector<float> vx(vertexCount);
	std::vector<float> vy(vertexCount);
	std::vector<float> vz(vertexCount);

	int iter = 0;
	std::for_each(vertices.begin(), vertices.end(), [&](auto& v)
		{
			vx[iter] = v.x;
			vy[iter] = v.y;
			vz[iter++] = v.z;
		});

	// copy
	HANDLE_ERROR(cudaMemcpy(indices, indexData.data(), 3 * triangleCount * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(positions.x, vx.data(), vertexCount * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(positions.y, vy.data(), vertexCount * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(positions.z, vz.data(), vertexCount * sizeof(float), cudaMemcpyHostToDevice));

	// centers
	int threadsPerBlock = triangleCount > 1024 ? 1024 : triangleCount;
	int blocks = std::ceil(static_cast<float>(triangleCount) / threadsPerBlock);
	calculateCenters(triangleCount > 1024 ? 1024 : triangleCount, std::ceil(static_cast<float>(triangleCount) / threadsPerBlock));
}

VeinTriangles::VeinTriangles(const VeinTriangles& other) : isCopy(true), triangleCount(other.triangleCount), vertexCount(other.vertexCount),
positions(other.positions), velocities(other.velocities), forces(other.forces), indices(other.indices), centers(other.centers),
veinVertexHorizontalDistance(other.veinVertexHorizontalDistance),
veinVertexNonHorizontalDistances{ other.veinVertexNonHorizontalDistances[0], other.veinVertexNonHorizontalDistances[1], other.veinVertexNonHorizontalDistances[2] }
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
	int vertex = blockDim.x * blockIdx.x + threadIdx.x;
	if (vertex >= triangles.vertexCount)
		return;

	float springForce;
	float3 neighborPosition;

	float3 vertexPosition = triangles.positions.get(vertex);
	float3 vertexVelocity = triangles.velocities.get(vertex);
	float3 vertexForce = { 0,0,0 };

	// Calculate our own spatial indices
	int i = vertex / cylinderHorizontalLayers;
	int j = vertex - i * cylinderHorizontalLayers;

	// vertically adjacent vertices

	int jSpan[] =
	{
		j != 0 ? j - 1 : cylinderHorizontalLayers - 1,
		j,
		(j + 1) % cylinderHorizontalLayers
	};


	int vertexHorizontalPrev = i * cylinderHorizontalLayers + jSpan[0];
	int vertexHorizontalNext = i * cylinderHorizontalLayers + jSpan[2];


	// Previous horizontally
	neighborPosition = triangles.positions.get(vertexHorizontalPrev);
	springForce = triangles.calculateVeinSpringForce(vertexPosition, neighborPosition, vertexVelocity, triangles.velocities.get(vertexHorizontalPrev),
		triangles.veinVertexHorizontalDistance);
	vertexForce = vertexForce + springForce * normalize(neighborPosition - vertexPosition);

	// Next horizontally
	neighborPosition = triangles.positions.get(vertexHorizontalNext);
	springForce = triangles.calculateVeinSpringForce(vertexPosition, neighborPosition, vertexVelocity, triangles.velocities.get(vertexHorizontalNext),
		triangles.veinVertexHorizontalDistance);
	vertexForce = vertexForce + springForce * normalize(neighborPosition - vertexPosition);



	// not the lower end of the vein
	if (i != 0)
	{
		// Lower vertical neighbors
#pragma unroll
		for (int jIndex = 0; jIndex < 3; jIndex++)
		{
			int vertexVerticalPrev = (i - 1) * cylinderHorizontalLayers + jSpan[jIndex];
			neighborPosition = triangles.positions.get(vertexVerticalPrev);
			springForce = triangles.calculateVeinSpringForce(vertexPosition, neighborPosition, vertexVelocity, triangles.velocities.get(vertexVerticalPrev),
				triangles.veinVertexNonHorizontalDistances[jIndex]);
			vertexForce = vertexForce + springForce * normalize(neighborPosition - vertexPosition);
		}

	}

	//// not the upper end of the vein
	if (i != cylinderVerticalLayers - 1)
	{
		// Upper vertical neighbors
#pragma unroll
		for (int jIndex = 0; jIndex < 3; jIndex++)
		{
			int vertexVerticalNext = (i + 1) * cylinderHorizontalLayers + jSpan[jIndex];
			neighborPosition = triangles.positions.get(vertexVerticalNext);
			springForce = triangles.calculateVeinSpringForce(vertexPosition, neighborPosition, vertexVelocity, triangles.velocities.get(vertexVerticalNext),
				triangles.veinVertexNonHorizontalDistances[jIndex]);
			vertexForce = vertexForce + springForce * normalize(neighborPosition - vertexPosition);
		}

	}

	triangles.forces.add(vertex, vertexForce);
}

/// <summary>
/// Gather forces from neighboring vertices, synchronize and then update forces for each vertex
/// </summary>
void VeinTriangles::gatherForcesFromNeighbors(int blocks, int threadsPerBlock)
{
	gatherForcesKernel << <blocks, threadsPerBlock >> > (*this);
}
