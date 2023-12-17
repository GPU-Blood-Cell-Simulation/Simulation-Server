#pragma once

// Frames to render
inline constexpr int maxFrames = 20000; //500;

inline constexpr int maxCudaStreams = 16;

// ! this value should be determined experimentally !
// one frame simulation time span
inline constexpr float dt = 0.008f;

// Blood flow
inline constexpr bool useBloodFlow = true;

// uniform grid cell count
inline constexpr int cellWidth = 20;
inline constexpr int cellHeight = 20;
inline constexpr int cellDepth = 20;

//inline constexpr int cellCountX = static_cast<int>(width / cellWidth);
//inline constexpr int cellCountY = static_cast<int>(height / cellHeight);
//inline constexpr int cellCountZ = static_cast<int>(depth / cellDepth);

// represents ratio of radius of maximal and actual bounding sphere
// for particles in cells
inline constexpr int boundingSpheresCoeff = 5;
inline constexpr float gridYMargin = 10.0f;

inline constexpr float gridXZMargin = 20.0f;

inline constexpr float minSpawnY = -20;
