#pragma once

inline constexpr int maxFrames = 600;
inline constexpr int maxCudaStreams = 16;
inline constexpr float dt = 0.008;
inline constexpr bool useBloodFlow = 1;
inline constexpr bool enableReactionForce = 1;
inline constexpr bool enableBigCellsBrake = 1;

inline constexpr int cellWidth = 2;
inline constexpr int cellHeight = 2;
inline constexpr int cellDepth  = 2;

inline constexpr int cellWidthTriangles = 25;
inline constexpr int cellHeightTriangles = 25;
inline constexpr int cellDepthTriangles = 25;

inline constexpr int boundingSpheresCoeff = 3;
inline constexpr float gridYMargin = 40;
inline constexpr float gridXZMargin = 40;
inline constexpr float minSpawnY = -20;
