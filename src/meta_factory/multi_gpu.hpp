#pragma once

#ifdef MULTI_GPU
    //GPU_COUNT_DEPENDENT
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
		accumulated += size / gpuCount;
	}

    return ret;
};

inline constexpr auto gpuSplitEndGenerator = [](int size)
{
	std::array<int, gpuCount> ret;

	int accumulated = 0;
	for (int i = 0; i < gpuCount - 1; i++)
	{
        accumulated += size / gpuCount;
		ret[i] = accumulated;		
	}
	ret[gpuCount - 1] = size;

    return ret;
};