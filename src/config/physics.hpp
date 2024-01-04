#pragma once
#include "simulation.hpp"

inline constexpr float velocity_collision_damping = 0.96;
inline constexpr float particle_k_sniff = 20;
inline constexpr float vein_k_sniff = 0.05;

inline constexpr float particle_d_fact = 9;
inline constexpr float vein_d_fact = 0.45;

inline constexpr float viscous_damping = 0.39;
inline constexpr float vein_boundaries_velocity_damping = 0.65;
inline constexpr float vein_collision_force_intensity = 0.08;

inline constexpr float collisionSpringCoeff = 6;
inline constexpr float collisionDampingCoeff = 6;
inline constexpr float collistionShearCoeff = 4;

inline constexpr float initVelocityX = 0;
inline constexpr float initVelocityY = -80;
inline constexpr float initVelocityZ = 0;
inline constexpr float randomVelocityModifier = 0.894;

inline constexpr float veinImpactDistance = 6;
inline constexpr float veinImpactMinimalForceDistance = 0.0001;

inline constexpr float Gx = 0;
inline constexpr float Gy = -25;
inline constexpr float Gz = 0;
