#pragma once

#include "constexpr_vec.hpp"

#include <cmath>
#include <cstdio>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__host__ __device__ inline float3 operator*(float a, float3 v)
{
	return make_float3(a * v.x, a * v.y, a * v.z);
}

__host__ __device__ inline float3 operator*(float3 a, float3 b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ inline float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator/(float3 v, float a)
{
	return make_float3(v.x / a, v.y / a, v.z / a);
}

/// <summary>
/// Checks if v is (0,0,0)
/// </summary>
/// <param name="v">input vector</param>
/// <returns>boolean value of this property</returns>
__host__ __device__ inline bool isEmpty(float3 v)
{
	return v.x == 0 && v.y == 0 && v.z == 0;
}

/// <summary>
/// Calculates dot product
/// </summary>
/// <param name="a">first vector</param>
/// <param name="b">second vector</param>
/// <returns>dot product value of a and b</returns>
__host__ __device__ inline float dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

/// <summary>
/// Calculates cross product
/// </summary>
/// <param name="a">first vector</param>
/// <param name="b">second vector</param>
/// <returns>cross product value of a and b</returns>
__host__ __device__ inline float3 cross(float3 u, float3 v)
{
	return make_float3(u.y * v.z - u.z * v.y,
		u.z * v.x - u.x * v.z,
		u.x * v.y - u.y * v.x);
}

/// <summary>
/// Calculate square of length of vector
/// </summary>
/// <param name="v">input vector</param>
/// <returns>length squared</returns>
__host__ __device__ inline float length_squared(float3 v)
{
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

/// <summary>
/// Calculate length of vector
/// </summary>
/// <param name="v">input vector</param>
/// <returns>length</returns>
__host__ __device__ inline float length(float3 v)
{
	return sqrtf(length_squared(v));
}

/// <summary>
/// Normalize given vector
/// </summary>
/// <param name="v">input</param>
/// <returns>normalized vector</returns>
__host__ __device__ inline float3 normalize(float3 v) // versor
{
	float3 vn = v / sqrtf(dot(v, v));
	if (isnan(vn.x) || isnan(vn.y) || isnan(vn.z))
	{
		printf("[NAN! vx=%.5f, vy=%.5f, vz=%.5f] ", v.x, v.y, v.z);
		return make_float3(0, 0, 0);
	}
	return vn;
}

/// <summary>
/// constexpr version of ceil function - normal one is notavaiable in constexpr functions
/// </summary>
/// <param name="num">argument of function ceil</param>
/// <returns></returns>
constexpr int constCeil(float num)
{
	return (static_cast<float>(static_cast<int>(num)) == num)
		? static_cast<int>(num)
		: static_cast<int>(num) + ((num > 0) ? 1 : 0);
}
