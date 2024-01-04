#pragma once

#include <cstdio>
#include <cstdlib>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


/// <summary>
/// A handy function for easy error checking
/// </summary>
/// <param name="err">cuda error</param>
/// <param name="file">file name</param>
/// <param name="line">line in file</param>
static void HandleError(cudaError_t err, const char* file, int line);

static void HandleError(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
            file, line);
        cudaDeviceSynchronize();
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ));