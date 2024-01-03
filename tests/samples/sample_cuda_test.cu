#include <gtest/gtest.h>

#include "sample_cuda_code/vector_add.cuh"


TEST(SampleTests, TestIfCudaIsWorks) {
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	const int exp_c[arraySize] = { 11, 22, 33, 44, 55 };
	int c[arraySize] = { 0 };
	
	// Add vectors in parallel.
	cudaError_t cudaStatus = sample_cuda::addWithCuda(c, a, b, arraySize);
	ASSERT_EQ(cudaStatus, cudaSuccess);

	for (int i = 0; i < arraySize; i++) {
		ASSERT_EQ(c[i], exp_c[i]);
	}
	
	cudaStatus = cudaDeviceReset();
	ASSERT_EQ(cudaStatus, cudaSuccess);
}