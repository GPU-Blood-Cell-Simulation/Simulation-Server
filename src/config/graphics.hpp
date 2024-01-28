#pragma once

#include <string>

inline int VEIN_POLYGON_MODE = 0;
inline bool BLOOD_CELL_SPRINGS_RENDER = true;
inline bool BLOOD_CELL_SPHERE_RENDER = false;

inline constexpr bool useLighting = true;

inline constexpr float cameraMovementSpeedCoefficient = 0.01;
inline constexpr float cameraRotationSpeed = 0.02;

inline constexpr int windowWidth = 800;
inline constexpr int windowHeight = 800;

inline constexpr bool saveVideoToFile = true;
inline std::string videoFileName = "blood_simulation.mov";