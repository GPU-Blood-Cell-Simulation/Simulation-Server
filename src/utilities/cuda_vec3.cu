#include "cuda_vec3.cuh"

#include "cuda_handle_error.cuh"

cudaVec3::cudaVec3(int n)
{
	HANDLE_ERROR(cudaMalloc((void**)&x, n * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&y, n * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&z, n * sizeof(float)));
}

cudaVec3::cudaVec3(const cudaVec3& other) : isCopy(true), x(other.x), y(other.y), z(other.z) {}

cudaVec3::~cudaVec3()
{
	if (!isCopy)
	{
		HANDLE_ERROR(cudaFree(x));
		HANDLE_ERROR(cudaFree(y));
		HANDLE_ERROR(cudaFree(z));
	}
}
