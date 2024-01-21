#pragma once

#include "cuda_vec3.cuh"

#include <tuple>
#include <type_traits>

#include "cuda_runtime.h"

template <typename T, size_t N, class = std::make_index_sequence<N>>
class HostDeviceArray;

/// <summary>
/// Cuda-friendly copyable stack array. Based on libstdc++ implementation
/// </summary>
/// <param name="T">Type of objects stored</param>
/// <param name="N">Array size</param>
template <typename T, size_t N, size_t... Is>
class HostDeviceArray<T, N, std::index_sequence<Is...>> 
{
private:
    T data[N];

    template <size_t >
    using pair_ = std::pair<int, int>;

    template <size_t >
    using T_ = T;
    
public:
    using DataType = T;

    __host__ __device__ constexpr HostDeviceArray() {}

    // Special constructor for cudaVec3 to avoid copies
    template<typename U = T>
    __host__ HostDeviceArray(pair_<Is>... args, typename std::enable_if<std::is_same<U, cudaVec3>::value>::type* = 0) : data{args...} { }

    template<typename U = T>
    __host__ HostDeviceArray(T_<Is>... args, typename std::enable_if<!std::is_same<U, cudaVec3>::value>::type* = 0) : data{args...} { }
    
    __host__ constexpr HostDeviceArray(const HostDeviceArray& other)
    {
        for (size_t i = 0; i < N; i++)
        {
            data[i] = other[i];
        }
    }

    __host__ inline constexpr HostDeviceArray& operator=(const HostDeviceArray& other)
    {
        for (size_t i = 0; i < N; i++)
        {
            data[i] = other[i];
        }
        return *this;
    }

    __host__ __device__ inline constexpr T& operator[](size_t __n)
    {
        return data[__n];
    }
    
    __host__ __device__ inline constexpr const T& operator[](size_t __n) const
    {
        return data[__n];
    }
};