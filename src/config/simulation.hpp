#pragma once

inline constexpr int maxFrames = 600;
inline constexpr int maxCudaStreams = 16;
inline constexpr float dt = 0.008;
inline constexpr bool useBloodFlow = 1;
inline constexpr bool enableReactionForce = 1;
inline constexpr bool enableBigCellsBrake = 1;

inline constexpr int cellWidth = 30;
inline constexpr int cellHeight = 30;
inline constexpr int cellDepth  = 30;

inline constexpr int boundingSpheresCoeff = 3;
inline constexpr float gridYMargin = 20;
inline constexpr float gridXZMargin = 20;
inline constexpr float minSpawnY = -20;
