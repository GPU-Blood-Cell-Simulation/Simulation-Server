#include <gtest/gtest.h>

#include "../../src/utilities/cuda_threads.hpp"


TEST(CudaThreadsTests, OneInstance) {
	CudaThreads threads(1);

	EXPECT_EQ(1, threads.blocks);
	EXPECT_EQ(1, threads.threadsPerBlock);
}


TEST(CudaThreadsTests, LessThanMaxThreads) {
	const int instances = CudaThreads::maxThreadsInBlock / 2;

	CudaThreads threads(instances);

	EXPECT_EQ(1, threads.blocks);
	EXPECT_EQ(instances, threads.threadsPerBlock);
}


TEST(CudaThreadsTests, MaxThreads) {
	CudaThreads threads(CudaThreads::maxThreadsInBlock);

	EXPECT_EQ(1, threads.blocks);
	EXPECT_EQ(CudaThreads::maxThreadsInBlock, threads.threadsPerBlock);
}


TEST(CudaThreadsTests, TwoTimesMaxThreads) {
	CudaThreads threads(CudaThreads::maxThreadsInBlock * 2);

	EXPECT_EQ(2, threads.blocks);
	EXPECT_EQ(CudaThreads::maxThreadsInBlock, threads.threadsPerBlock);
}


TEST(CudaThreadsTests, TwoAndAHalfTimesMaxThreads) {
	CudaThreads threads(CudaThreads::maxThreadsInBlock * 2.5);

	EXPECT_EQ(3, threads.blocks);
	EXPECT_EQ(CudaThreads::maxThreadsInBlock, threads.threadsPerBlock);
}
