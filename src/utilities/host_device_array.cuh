#pragma once

#include "cuda_runtime.h"

/// <summary>
/// Cuda-friendly copyable stack array. Based on libstdc++ implementation
/// </summary>
/// <param name="T">Type of objects stored</param>
/// <param name="N">Array size</param>
template<typename T, int N>
struct HostDeviceArray
{
    __host__ __device__ constexpr HostDeviceArray() {}

    __host__ __device__ constexpr HostDeviceArray(const HostDeviceArray<T, N>& other)
    {
        for (int i = 0; i < N; i++)
        {
            _M_instance[i] = other[i];
        }
    }

    __host__ __device__ inline constexpr HostDeviceArray<T, N>& operator=(const HostDeviceArray<T, N>& other)
    {
        for (int i = 0; i < N; i++)
        {
            _M_instance[i] = other[i];
        }
        return *this;
    }

    __host__ __device__ inline constexpr T& operator[](size_t __n)
    {
        return _M_instance[__n];
    }
    
    __host__ __device__ inline constexpr const T& operator[](size_t __n) const
    {
        return _M_instance[__n];
    }

    private:
        T _M_instance[N];
};