#include <gtest/gtest.h>

#include "../../src/utilities/cuda_vec3.cuh"


class CudaVec3Tests: public testing::TestWithParam<int /* Vector Len */> {
public:
    bool *device_test_result;

    void SetUp() override {
        ASSERT_EQ(cudaMalloc(&device_test_result, sizeof(bool)), cudaSuccess);
    }

    bool GetTestResult() {
        bool host_test_result;
        cudaMemcpy(&host_test_result, device_test_result, sizeof(bool), cudaMemcpyDeviceToHost); 
        return host_test_result;
    }

    void TearDown() override {
        cudaFree(device_test_result);
    }
};


INSTANTIATE_TEST_CASE_P(ZeroLenCudaVector, CudaVec3Tests, testing::Values(0));
INSTANTIATE_TEST_CASE_P(ShortCudaVector, CudaVec3Tests, testing::Range(1, 5));
INSTANTIATE_TEST_SUITE_P(LongCudaVector, CudaVec3Tests, testing::Values(100, 500, 1000, 10000));


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

TEST_P(CudaVec3Tests, GetAndSet) {
    int n = GetParam();
    cudaVec3 vec(n);

    cudaVec3GetAndSetCheck<<<1, 1>>>(vec, n, device_test_result);
    
    EXPECT_TRUE(GetTestResult());
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

TEST_P(CudaVec3Tests, Addition) {
    int n = GetParam();
    cudaVec3 vec(n);

    cudaVec3AddCheck<<<1, 1>>>(vec, n, device_test_result);

    EXPECT_TRUE(GetTestResult());
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

TEST_P(CudaVec3Tests, Copy) {
    int n = GetParam();
    cudaVec3 vec1(n);
    cudaVec3 vec2(vec1);

    cudaVec3IdentityCheck<<<1, 1>>>(vec1, vec2, n, device_test_result);

    EXPECT_TRUE(GetTestResult());
}
