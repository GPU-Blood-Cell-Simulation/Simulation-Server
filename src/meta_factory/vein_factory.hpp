#pragma once

#include "../config/simulation.hpp"
#include "../config/vein_definition.hpp"
#include <algorithm>

inline constexpr int veinPositionCount = veinPositions.size();
inline constexpr int veinIndexCount = veinIndices.size();

inline constexpr auto calculateMin = [](int dim)
	{
		cvec vec = *std::min_element(veinPositions.begin(), veinPositions.end(), [&](auto v1, auto v2)
			{
				return v1[dim] < v2[dim];
			});
		return vec[dim] - (dim == 1 ? gridYMargin : gridXZMargin);
	};

inline constexpr auto calculateMax = [](int dim)
	{
		cvec vec = *std::max_element(veinPositions.begin(), veinPositions.end(), [&](auto v1, auto v2)
			{
				return v1[dim] < v2[dim];
			});
		return vec[dim] + (dim == 1 ? gridYMargin : gridXZMargin);
	};

inline constexpr float minX = calculateMin(0);
inline constexpr float maxX = calculateMax(0);
inline constexpr float maxY = calculateMax(1);
inline constexpr float minY = calculateMin(1);
inline constexpr float maxZ = calculateMax(2);
inline constexpr float minZ = calculateMin(2);

inline constexpr float width = maxX - minX;
inline constexpr float height = maxY - minY;
inline constexpr float depth = maxZ - minZ;

// TODO: read hLayers
inline constexpr float cylinderRadius = 50;