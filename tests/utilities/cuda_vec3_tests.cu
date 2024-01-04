#include <gtest/gtest.h>

#include "../../src/utilities/cuda_vec3.cuh"


__global__ void cudaVec3GetAndSetCheck(cudaVec3 vec, int n, bool* result) {
    *result = true;
    
    for (int i = 0; i < n; i++) {
        float3 f = make_float3(i, i+1, i+2);
        vec.set(i, f);
    }

    for (int i = 0; i < n; i++) {
        if (vec.get(i).x != i ||
            vec.get(i).y != i+1 ||
            vec.get(i).z != i+2)
        {
            *result = false;
            return;
        }
    }
}

TEST(CudaVec3, GetAndSet) {
    bool h_result;
    bool* d_result;
    int n = 5;
    cudaVec3 vec(n);

    cudaMalloc(&d_result, sizeof(bool));

    cudaVec3GetAndSetCheck<<<1, 1>>>(vec, n, d_result);

    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost); 
    cudaFree(d_result);
    
    EXPECT_TRUE(h_result);
}


__global__ void cudaVec3AddCheck(cudaVec3 vec, int n, bool* result) {
    *result = true;
    
    for (int i = 0; i < n; i++) {
        float3 f = make_float3(i, i+1, i+2);
        vec.set(i, f);
        vec.add(i, f);
    }

    for (int i = 0; i < n; i++) {
        if (vec.get(i).x != 2*i ||
            vec.get(i).y != 2*(i+1) ||
            vec.get(i).z != 2*(i+2))
        {
            *result = false;
            return;
        }
    }
}

TEST(CudaVec3, Addition) {
    bool h_result;
    bool* d_result;
    int n = 5;
    cudaVec3 vec(n);

    cudaMalloc(&d_result, sizeof(bool));

    cudaVec3AddCheck<<<1, 1>>>(vec, n, d_result);

    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost); 
    cudaFree(d_result);
    
    EXPECT_TRUE(h_result);
}


__global__ void cudaVec3IdentityCheck(cudaVec3 vec1, cudaVec3 vec2, int n, bool* result) {
    *result = true;

    for (int i = 0; i < n; i++) {
        if (vec1.get(i).x != vec2.get(i).x ||
            vec1.get(i).y != vec2.get(i).y ||
            vec1.get(i).z != vec2.get(i).z )
        {
            *result = false;
            return;
        }
    }
}

TEST(CudaVec3, Copy) {
    bool h_result;
    bool* d_result;
    int n = 5;
    cudaVec3 vec1(n);
    cudaVec3 vec2(vec1);

    cudaMalloc(&d_result, sizeof(bool));

    cudaVec3IdentityCheck<<<1, 1>>>(vec1, vec2, n, d_result);

    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost); 
    cudaFree(d_result);
    
    EXPECT_TRUE(h_result);
}
