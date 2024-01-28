#pragma once

#include <tuple>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/// <summary>
/// Represents 3 float vectors on gpu. Uses RAII to handle memory management
/// </summary>
class cudaVec3
{
	bool isCopy = true;
	int gpuId = 0;

public:
	float* x = 0;
	float* y = 0;
	float* z = 0;

	// allocated on host
	cudaVec3() = default;

	/// <summary>
	/// Basic constructor
	/// </summary>
	/// <param name="n">vector size</param>
	/// <param name="gpuId">the gpu where the vector memory should be allocated</param>
	cudaVec3(int n, int gpuId = 0);

	cudaVec3(std::pair<int, int> pair);

	cudaVec3(const cudaVec3& other);
	cudaVec3& operator=(const cudaVec3& other);

	~cudaVec3();

	/// <summary>
	/// Gets certaing vector as float3
	/// </summary>
	/// <param name="index">vector index</param>
	/// <returns>vector as float3</returns>
	__device__ inline float3 get(int index) const
	{
		return make_float3(x[index], y[index], z[index]);
	}

	/// <summary>
	/// Sets certain vector
	/// </summary>
	/// <param name="index">vector index</param>
	/// <param name="v">vector value</param>
	/// <returns></returns>
	__device__ inline void set(int index, float3 v)
	{
		x[index] = v.x;
		y[index] = v.y;
		z[index] = v.z;
	}

	/// <summary>
	/// Adds value to the certaing vector
	/// </summary>
	/// <param name="index">vector index</param>
	/// <param name="v">vector value</param>
	/// <returns></returns>
	__device__ inline void add(int index, float3 v)
	{
		x[index] += v.x;
		y[index] += v.y;
		z[index] += v.z;
	}
};