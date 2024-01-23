#pragma once

#include <cstdio>
#include <cstdlib>

#include "cuda_runtime.h"
#ifdef MULTI_GPU
#include "nccl.h"
#endif


/// <summary>
/// A handy function for easy error checking
/// </summary>
/// <param name="err">cuda error</param>
/// <param name="file">file name</param>
/// <param name="line">line in file</param>
#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#ifdef MULTI_GPU
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0);
#endif
