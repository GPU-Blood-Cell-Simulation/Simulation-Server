#pragma once

#ifdef MULTI_GPU
	inline constexpr int gpuCount = 3;
	inline constexpr int particleGridGpu = 0;
	inline constexpr int veinGridGpu = 0;
#else
	inline constexpr int gpuCount = 1;
	inline constexpr int particleGridGpu = 0;
	inline constexpr int veinGridGpu = 0;
#endif

// Multi gpu split lambdas
inline constexpr auto gpuSplitSizeGenerator = [](int size)
{
	std::array<int, gpuCount> ret;

	int accumulated = 0;
	for (int i = 0; i < gpuCount - 1; i++)
	{
		ret[i] = i * size / gpuCount;
		accumulated += i * size / gpuCount;
	}
	if (gpuCount > 1)
	{
		ret[gpuCount- 1] = size - accumulated;
	}

    return ret;
};

inline constexpr auto gpuSplitOffsetGenerator = [](int size)
{
	std::array<int, gpuCount> ret;

	int accumulated = 0;
	for (int i = 0; i < gpuCount; i++)
	{
		ret[i] = accumulated;
		accumulated += i * size / gpuCount;
	}

    return ret;
};