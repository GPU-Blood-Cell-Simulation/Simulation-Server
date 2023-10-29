#pragma once

// Blood flow
inline constexpr bool useBloodFlow = true;

// factor to slow down particles after collision
inline constexpr float velocityCollisionDamping = 0.8f;

// ! this value should be determined experimentally !
// Hooks law k factor from F = k*x
inline constexpr float particle_k_sniff = 0.1f;
inline constexpr float vein_k_sniff = 0.3f;

// ! this value should be determined experimentally !
// Damping factor 
inline constexpr float particle_d_fact = 0.1f;
inline constexpr float vein_d_fact = 0;

// Particle-particle collision coefficients
inline constexpr float collisionSpringCoeff = 0.2f;
inline constexpr float collisionDampingCoeff = 0.02f;
inline constexpr float collistionShearCoeff = 0.05f;

// Initial velocity of particles
inline constexpr float initVelocityX = 0.0f;
inline constexpr float initVelocityY = -10.0f;
inline constexpr float initVelocityZ = 0.0f;

// distance beetwen particle and vein on which an impact occurs
inline constexpr float veinImpactDistance = 5.0f;