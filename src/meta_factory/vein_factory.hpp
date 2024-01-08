#pragma once

#include "../config/simulation.hpp"
#include "../config/vein_definition.hpp"
#include "../utilities/constexpr_vec.hpp"
#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

inline constexpr int veinPositionCount = veinPositions.size();
inline constexpr int veinIndexCount = veinIndices.size();
inline constexpr int triangleCount = veinIndexCount / 3;

namespace internal
{
	/// <summary>
	/// Calculates lower bound for vein grid
	/// </summary>
	inline constexpr auto calculateMin = [](int dim)
	{
		cvec vec = *std::min_element(veinPositions.begin(), veinPositions.end(), [&](auto v1, auto v2)
			{
				return v1[dim] < v2[dim];
			});
		return vec[dim] - (dim == 1 ? gridYMargin : gridXZMargin);
	};

	/// <summary>
	/// Calculates upper bound for vein grid
	/// </summary>
	inline constexpr auto calculateMax = [](int dim)
	{
		cvec vec = *std::max_element(veinPositions.begin(), veinPositions.end(), [&](auto v1, auto v2)
			{
				return v1[dim] < v2[dim];
			});
		return vec[dim] + (dim == 1 ? gridYMargin : gridXZMargin);
	};
}

/// <summary>
/// X lower bound of vein grid
/// </summary>
inline constexpr float minX = internal::calculateMin(0);

/// <summary>
/// X upper bound of vein grid
/// </summary>
inline constexpr float maxX = internal::calculateMax(0);

/// <summary>
/// Y upper bound of vein grid
/// </summary>
inline constexpr float maxY = internal::calculateMax(1);

/// <summary>
/// Y lower bound of vein grid
/// </summary>
inline constexpr float minY = internal::calculateMin(1);

/// <summary>
/// Z upper bound of vein grid
/// </summary>
inline constexpr float maxZ = internal::calculateMax(2);

/// <summary>
/// Z lower bound of vein grid
/// </summary>
inline constexpr float minZ = internal::calculateMin(2);

/// <summary>
/// Grid's width
/// </summary>
inline constexpr float width = maxX - minX;

/// <summary>
/// Grid's height
/// </summary>
inline constexpr float height = maxY - minY;

/// <summary>
/// Grid's depth
/// </summary>
inline constexpr float depth = maxZ - minZ;

// TODO: read hLayers
inline constexpr float cylinderRadius = 50;

namespace internal
{
	inline float length_squared(cvec v)
	{
		return v.x * v.x + v.y * v.y + v.z * v.z;
	}
	inline float length(cvec v)
	{
		return sqrtf(length_squared(v));
	}

	template<typename T>
	inline float abs(T v)
	{
		return v < 0 ? -v : v;
	}

	/// <summary>
	/// Calculate maximum number of neighbors per vein vertex
	/// </summary>
	inline constexpr auto calculateMaxNeighbors = []()
	{
		std::array<int, veinPositionCount> neighborCount{0};
		for (unsigned int i = 0; i < veinIndexCount; i+=3)
		{
			neighborCount[veinIndices[i]]++;
			neighborCount[veinIndices[i + 1]]++;
			neighborCount[veinIndices[i + 2]]++;
		}

		return *std::max(neighborCount.begin(), neighborCount.end());
	};
}

// TODO: Hardcoded for now because the lambda above is too difficult for gcc, perhaps there is another way?
inline constexpr int veinVertexMaxNeighbors = 9; //internal::calculateMaxNeighbors();

/// <summary>
/// Calculate vein neighbor lists for cuda
/// </summary>
inline auto calculateSpringLengths = []()
{
	std::array<std::vector<unsigned int>, veinPositionCount> neighbors {};
	std::array<std::vector<int>, veinVertexMaxNeighbors> generatedNeighborArrays {};
	std::array<std::vector<float>, veinVertexMaxNeighbors> springLengths {};

	// Fill neighbor vectors
	for (unsigned int i = 0; i < veinIndexCount; i+=3)
	{
		auto i0 = veinIndices[i];
		auto i1 = veinIndices[i + 1];
		auto i2 = veinIndices[i + 2];
		neighbors[i0].push_back(i1);
		neighbors[i0].push_back(i2);
		neighbors[i1].push_back(i0);
		neighbors[i1].push_back(i2);
		neighbors[i2].push_back(i0);
		neighbors[i2].push_back(i1);
	}

	for (unsigned int i = 0; i < veinPositionCount; i++)
	{
		// Heuristic - sort the vector. In most cases this ensures better coalescing in cuda kernels
		std::sort(neighbors[i].begin(), neighbors[i].end());
	}

	for (unsigned int i = 0; i < veinVertexMaxNeighbors; i++)
	{
		generatedNeighborArrays[i] = std::vector(veinPositionCount, -1);
		springLengths[i] = std::vector(veinPositionCount, -1.0f);
	}

	// Split into separate arrays and calculate distances
	for (unsigned int i = 0; i < veinPositionCount; i++)
	{
		for (unsigned int j = 0; j < std::min<size_t>(neighbors[i].size(), veinVertexMaxNeighbors); j++)
		{
			auto neighbor = neighbors[i][j];
			generatedNeighborArrays[j][i] = neighbor;
			springLengths[j][i] = internal::abs(internal::length(veinPositions[i] - veinPositions[neighbor]));
		}
	}

	return std::tuple(std::move(generatedNeighborArrays), std::move(springLengths));
};

inline constexpr int veinEndingCenterCount = mp_size<VeinEndingCenters>::value;
inline constexpr int veinEndingRadiusCount = mp_size<VeinEndingRadii>::value;
static_assert(veinEndingCenterCount == veinEndingRadiusCount, "Ill-formed vein ednings");