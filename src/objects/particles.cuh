#pragma once

#include "../meta_factory/blood_cell_factory.hpp"
#include "../utilities/cuda_vec3.cuh"
#include "../utilities/host_device_array.cuh"

/// <summary>
/// A structure containing the position, velocity and force vectors of all particles
/// </summary>
struct Particles
{
	#ifdef MULTI_GPU
	HostDeviceArray<cudaVec3, gpuCount> positions { //GPU_COUNT_DEPENDENT
		{particleCount, 0},
		{particleCount, 1}, 
		{particleCount, 2},
		{particleCount, 3}
	};
	HostDeviceArray<cudaVec3, gpuCount> velocities { //GPU_COUNT_DEPENDENT
		{particleCount, 0},
		{particleCount, 1}, 
		{particleCount, 2},
		{particleCount, 3}
	};
	HostDeviceArray<cudaVec3, gpuCount> forces { //GPU_COUNT_DEPENDENT
		{particleCount, 0},
		{particleCount, 1}, 
		{particleCount, 2},
		{particleCount, 3}
	};
	#else
	HostDeviceArray<cudaVec3, gpuCount> positions {{particleCount, 0}};
	HostDeviceArray<cudaVec3, gpuCount> velocities {{particleCount, 0}};
	HostDeviceArray<cudaVec3, gpuCount> forces {{particleCount, 0}};
	#endif
};