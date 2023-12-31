#pragma once
#include "simulation.hpp"

// factor to slow down particles after collision
inline constexpr float velocity_collision_damping = 0.96f;

// ! this value should be determined experimentally !
// Hooks law k factor from F = k*x
inline constexpr float particle_k_sniff = 20.0f; //1.0f/dt/dt;
inline constexpr float vein_k_sniff = 0.05f; 

// Damping factor. This value should be about sqrt(4*particle_k_sniff)
inline constexpr float particle_d_fact = 9.0f; //1.0f/dt;
inline constexpr float vein_d_fact = 0.45f;

// viscous damping slows down particle proportionally to its velocity
inline constexpr float viscous_damping = 0.3f; 

// suppression coefficient to slow down particles on vein boundaries
inline constexpr float vein_boundaries_velocity_damping = 0.65f;

// scale factor to calculate force applied to vein on collision with particle
inline constexpr float vein_collision_force_intensity = 0.08f;

// Particle-particle collision coefficients
inline constexpr float collisionSpringCoeff = 6.0f;
inline constexpr float collisionDampingCoeff = 6.0f;
inline constexpr float collistionShearCoeff = 4.0f;

// Initial velocity of particlesvariant
inline constexpr float initVelocityX = 0.0f;
inline constexpr float initVelocityY = -80.0f;
inline constexpr float initVelocityZ = 0.0f;
inline constexpr float randomVelocityModifier = 0.894f;


// distance beetwen particle and vein on which an impact occurs
inline constexpr float veinImpactDistance = 6.0f;
inline constexpr float veinImpactMinimalForceDistance = 0.0001f;

// gravity power
inline constexpr float Gx = 0.0f;
inline constexpr float Gy = -25.0f;
inline constexpr float Gz = 0.0f;