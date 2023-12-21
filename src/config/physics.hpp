#pragma once

// factor to slow down particles after collision
inline constexpr float velocityCollisionDamping = 0.8f;

// ! this value should be determined experimentally !
// Hooks law k factor from F = k*x
inline constexpr float particle_k_sniff = 20.0f;
inline constexpr float vein_k_sniff = 0.3f;

// ! this value should be determined experimentally !
// Damping factor 
inline constexpr float particle_d_fact = 20.0f;
inline constexpr float vein_d_fact = 0.0f;

inline constexpr float viscous_damping = 0.1f;
inline constexpr float vein_boundaries_velocity_damping = 0.65f;
// Particle-particle collision coefficients
inline constexpr float collisionSpringCoeff = 6.0f;
inline constexpr float collisionDampingCoeff = 6.0f;
inline constexpr float collistionShearCoeff = 4.0f;

// Initial velocity of particles
inline constexpr float initVelocityX = 0.0f;
inline constexpr float initVelocityY = -80.0f;
inline constexpr float initVelocityZ = 0.0f;
inline constexpr float randomVelocityModifier = 0.894f;


// distance beetwen particle and vein on which an impact occurs
inline constexpr float veinImpactDistance = 3.0f;
inline constexpr float veinImpactMinimalForceDistance = 0.0001f;

// gravity power
inline constexpr float Gx = 0.0f;
inline constexpr float Gy = -10.0f;
inline constexpr float Gz = 0.0f;