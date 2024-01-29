#pragma once

#include "cuda_handle_error.cuh"
#include "../meta_factory/blood_cell_factory.hpp"
#include "cuda_vec3.cuh"
#include "host_device_array.cuh"

#include <type_traits>

#include <nccl.h>
#include "cuda_runtime.h"


namespace nccl
{
    /// <summary>
	/// Synchronize all devices
	/// </summary>
	/// <param name="streams">NCCL synchronization stream array</param>
    inline void sync(cudaStream_t* streams)
    {
        for (int i = 0; i < gpuCount; i++)
        {
            CUDACHECK(cudaSetDevice(i));
            CUDACHECK(cudaStreamSynchronize(streams[i]));
        }
        CUDACHECK(cudaSetDevice(0));
    }

    /// <summary>
	/// Broadcast the provided data from the root node to all gpus
	/// </summary>
	/// <param name="data">data to broadcast</param>
	/// <param name="size">size of the data</param>
    /// <param name="type">NCCL data type</param>
    /// <param name="blocks">NCCL comm array</param>
	/// <param name="blocks">NCCL synchronization stream array</param>
    /// <param name="root">root gpu id</param>
    template<typename T>
    inline void broadcast(T& data, int size, ncclDataType_t type, ncclComm_t* comms, cudaStream_t* streams, int root = 0)
    {
        for (int i = 0; i < gpuCount; i++)
        {
            if constexpr (std::is_same<typename T::DataType, cudaVec3>::value)
            {
                NCCLCHECK(ncclBroadcast((const void*)data[i].x, (void*)data[i].x, size, type, root, comms[i], streams[i]));
                NCCLCHECK(ncclBroadcast((const void*)data[i].y, (void*)data[i].y, size, type, root, comms[i], streams[i]));
                NCCLCHECK(ncclBroadcast((const void*)data[i].z, (void*)data[i].z, size, type, root, comms[i], streams[i]));
            }
            else
            {
                NCCLCHECK(ncclBroadcast((const void*)data[i], (void*)data[i], size, type, root, comms[i], streams[i]));
            }  
        }   
    }

    /// <summary>
	/// Reduce provided data from all gpus into the root node
	/// </summary>
	/// <param name="data">data to reduce</param>
	/// <param name="size">size of the data</param>
    /// <param name="type">NCCL data type</param>
    /// <param name="blocks">NCCL comm array</param>
	/// <param name="blocks">NCCL synchronization stream array</param>
    template<typename T>
    inline void reduce(T& data, int size, ncclDataType_t type, ncclComm_t* comms, cudaStream_t* streams)
    {
        for (int i = 0; i < gpuCount; i++)
        {
            if constexpr (std::is_same<typename T::DataType, cudaVec3>::value)
            {
                NCCLCHECK(ncclReduce((const void*)data[i].x, (void*)data[i].x, size, type, ncclSum, 0, comms[i], streams[i]));
                NCCLCHECK(ncclReduce((const void*)data[i].y, (void*)data[i].y, size, type, ncclSum, 0, comms[i], streams[i]));
                NCCLCHECK(ncclReduce((const void*)data[i].z, (void*)data[i].z, size, type, ncclSum, 0, comms[i], streams[i]));
            }
            else
            {
                NCCLCHECK(ncclReduce((const void*)data[i], (void*)data[i], size, type, ncclSum, 0, comms[i], streams[i]));
            }  
        } 
    }
}