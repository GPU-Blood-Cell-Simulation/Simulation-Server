#pragma once

// Frames to render
inline constexpr int maxFrames = 500;

inline constexpr int maxCudaStreams = 16;

inline constexpr float width = 300.0f;
inline constexpr float height = 500.0f;
inline constexpr float depth = 300.0f;

// ! this value should be determined experimentally !
// one frame simulation time span
inline constexpr float dt = 0.016f;

// uniform grid cell count

inline constexpr int cellWidth = 20;
inline constexpr int cellHeight = 20;
inline constexpr int cellDepth = 20;

inline constexpr int cellCountX = static_cast<int>(width / cellWidth);
inline constexpr int cellCountY = static_cast<int>(height / cellHeight);
inline constexpr int cellCountZ = static_cast<int>(depth / cellDepth);