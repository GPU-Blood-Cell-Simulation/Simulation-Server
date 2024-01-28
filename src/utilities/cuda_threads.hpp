#pragma once

#include <cmath>
#include <stdexcept>

/// <summary>
/// Represents blocks and threads number for cuda kernels
/// </summary>
class CudaThreads
{
public:
	static constexpr int maxThreadsInBlock = 720;//1024;
	static constexpr int threadsInWarp = 32;

	const int threadsPerBlock;
	const int blocks;

	CudaThreads(int instances) :
		threadsPerBlock(instances > maxThreadsInBlock ? maxThreadsInBlock : instances),
		blocks(std::ceil(static_cast<float>(instances) / threadsPerBlock))
	{}

	CudaThreads(unsigned int threadsPerBlock, unsigned int blocks) :
		threadsPerBlock(threadsPerBlock), blocks(blocks)
	{}
};