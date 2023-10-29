#pragma once
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

__host__ __device__ inline bool isEmpty(float3 v)
{
	return v.x == 0 && v.y == 0 && v.z == 0;
}

__host__ __device__ inline float dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float3 cross(float3 u, float3 v)
{
	return make_float3(u.y * v.z - u.z * v.y,
		u.z * v.x - u.x * v.z,
		u.x * v.y - u.y * v.x);
}

__host__ __device__ inline float length_squared(float3 v)
{
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

__host__ __device__ inline float length(float3 v)
{
	return sqrtf(length_squared(v));
}

// TODO: is there a way to compute the inverse square root faster?
__host__ __device__ inline float3 normalize(float3 v) // versor
{
	float3 vn = v / sqrtf(dot(v, v));
	if (isnan(vn.x) || isnan(vn.y) || isnan(vn.z))
	{
		printf("NAN ");
	}
	return vn;
}

// constexpr version of ceil function - normal one is notavaiable in constexpr functions
constexpr int constCeil(float num)
{
	return (static_cast<float>(static_cast<int>(num)) == num)
		? static_cast<int>(num)
		: static_cast<int>(num) + ((num > 0) ? 1 : 0);
}
