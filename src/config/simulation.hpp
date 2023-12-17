#pragma once

// Frames to render
inline constexpr int maxFrames = 500;

inline constexpr int maxCudaStreams = 16;

// ! this value should be determined experimentally !
// one frame simulation time span
inline constexpr float dt = 0.002f;

inline constexpr float gridYMargin = 10.0f;

inline constexpr float gridXZMargin = 20.0f;

inline constexpr float minSpawnY = -20;