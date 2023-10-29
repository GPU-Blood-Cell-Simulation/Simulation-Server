#pragma once

#include "../config/simulation.hpp"
#include "../config/vein_definition.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

// cylinder model data:
inline constexpr float cylinderHeight = 0.8 * height;
inline constexpr float cylinderRadius = 0.2 * width;
inline glm::vec3 cylinderBaseCenter = glm::vec3(width / 2.0f, 0.1f * height, depth / 2.0f);
inline constexpr int cylinderVerticalLayers = 100;
inline constexpr int cylinderHorizontalLayers = 30;

inline constexpr int veinHeight = static_cast<int>(cylinderHeight);
inline constexpr int veinRadius = static_cast<int>(cylinderRadius);