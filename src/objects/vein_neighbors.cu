#include "vein_neighbors.cuh"

#include "../utilities/math.cuh"
#include "../utilities/cuda_handle_error.cuh"

#include <iostream>

VeinNeighbors::VeinNeighbors()
{
    const auto pair = calculateSpringLengths();
    const auto hostNeighbors = std::get<0>(pair);
    const auto hostSprings = std::get<1>(pair);

    int i = 0;
    for (auto&& pair : data)
    {   
        // Copy neighbor indices
        HANDLE_ERROR(cudaMalloc((void**)&pair.ids, veinPositionCount * sizeof(int)));
        HANDLE_ERROR(cudaMemcpy(pair.ids, hostNeighbors[i].data(), veinPositionCount * sizeof(int), cudaMemcpyHostToDevice));

        // Copy spring lengths
        HANDLE_ERROR(cudaMalloc((void**)&pair.springs, veinPositionCount * sizeof(float)));
        HANDLE_ERROR(cudaMemcpy(pair.springs, hostSprings[i].data(), veinPositionCount * sizeof(float), cudaMemcpyHostToDevice));
        i++;
    }
}

VeinNeighbors::VeinNeighbors(const VeinNeighbors& other) : isCopy(true)
{
    std::copy(other.data, other.data + veinVertexMaxNeighbors, data);
}

VeinNeighbors::~VeinNeighbors()
{
	if (isCopy)
        return;

    for (auto& [ids, springs] : data)
    {
        HANDLE_ERROR(cudaFree(ids));
        HANDLE_ERROR(cudaFree(springs));
    }
        
}
