#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//for __syncthreads()
//#ifndef __CUDACC_RTC__ 
//#define __CUDACC_RTC__
//#endif // !(__CUDACC_RTC__)
////for atomicAdd
//#ifndef __CUDACC__
//#define __CUDACC__
//#endif // !__CUDACC__

/// <summary>
/// Represents 3 float vectors on gpu. Uses RAII to handle memory management
/// </summary>
class cudaVec3
{
	bool isCopy = false;

public:
	float* x = 0;
	float* y = 0;
	float* z = 0;

	// allocated on host
	cudaVec3(int n);
	cudaVec3(const cudaVec3& other);
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
	/// Sets certaing vector
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

	/// !!! STILL NOT IMPLEMENTED !!!
	__device__ inline void atomicAddVec3(int index, float3 v)
	{
		/*atomicAdd(&x[index], v.x);
		atomicAdd(&y[index], v.z);
		atomicAdd(&z[index], v.y);*/
	}
};