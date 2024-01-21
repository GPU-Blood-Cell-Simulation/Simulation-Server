#include "cuda_vec3.cuh"

#include "cuda_handle_error.cuh"

cudaVec3::cudaVec3(int n, int gpuId) : gpuId(gpuId), isCopy(false)
{
	int previousDevice;
	CUDACHECK(cudaGetDevice(&previousDevice));
	CUDACHECK(cudaSetDevice(gpuId));

	CUDACHECK(cudaMalloc((void**)&x, n * sizeof(float)));
	CUDACHECK(cudaMalloc((void**)&y, n * sizeof(float)));
	CUDACHECK(cudaMalloc((void**)&z, n * sizeof(float)));

	CUDACHECK(cudaSetDevice(previousDevice));
}

cudaVec3::cudaVec3(std::pair<int, int> pair) : cudaVec3(pair.first, pair.second) {}

cudaVec3::cudaVec3(const cudaVec3& other) : isCopy(true), x(other.x), y(other.y), z(other.z) {}

cudaVec3& cudaVec3::operator=(const cudaVec3& other)
{
	isCopy = true;
	x = other.x;
	y = other.y;
	z = other.z;
	return *this;
}

cudaVec3::~cudaVec3()
{
	if (isCopy)
		return;
	
	int previousDevice;
	CUDACHECK(cudaGetDevice(&previousDevice));
	CUDACHECK(cudaSetDevice(gpuId));

	CUDACHECK(cudaFree(x));
	CUDACHECK(cudaFree(y));
	CUDACHECK(cudaFree(z));

	CUDACHECK(cudaSetDevice(previousDevice));
	
}
