#pragma once

#ifdef MULTI_GPU
	inline constexpr int gpuCount = 3;
	inline constexpr int particleGridGpu = 1;
	inline constexpr int veinGridGpu = 2;
#else
	inline constexpr int gpuCount = 1;
	inline constexpr int particleGridGpu = 0;
	inline constexpr int veinGridGpu = 0;
#endif

// Multi gpu split lambdas
inline constexpr auto gpuSplitStartGenerator = [](int size)
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

inline constexpr auto gpuSplitEndGenerator = [](int size)
{
	std::array<int, gpuCount> ret;

	int accumulated = 0;
	for (int i = 0; i < gpuCount - 1; i++)
	{
        accumulated += i * size / gpuCount;
		ret[i] = i * size / gpuCount;		
	}
	if (gpuCount > 1)
	{
		ret[gpuCount- 1] = size;
	}

    return ret;
};