#include "cuda_vec3.cuh"

#include "cuda_handle_error.cuh"

cudaVec3::cudaVec3(int n)
{
	CUDACHECK(cudaMalloc((void**)&x, n * sizeof(float)));
	CUDACHECK(cudaMalloc((void**)&y, n * sizeof(float)));
	CUDACHECK(cudaMalloc((void**)&z, n * sizeof(float)));
}

cudaVec3::cudaVec3(const cudaVec3& other) : isCopy(true), x(other.x), y(other.y), z(other.z) {}

cudaVec3::~cudaVec3()
{
	if (!isCopy)
	{
		CUDACHECK(cudaFree(x));
		CUDACHECK(cudaFree(y));
		CUDACHECK(cudaFree(z));
	}
}
