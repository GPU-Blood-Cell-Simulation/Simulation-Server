#pragma once

#include "../objects/particles.cuh"

/// <summary>
/// Base grid class - uses the Curiously Recurring Template Pattern to provide compile-time inheritance. Use it with std::variant and std::visit
/// 
/// </summary>
/// <typeparam name="Derived">A concrete grid implementation</typeparam>
template<typename Derived>
class BaseGrid
{
protected:
    BaseGrid() {}
public:


    inline void calculateGrid(const Particles& particles)
    {
        static_cast<Derived*>(this)->calculateGrid(particles);
    }
    inline void calculateGrid(const float* positionX, const float* positionY, const float* positionZ, int objectCount)
    {
        static_cast<Derived*>(this)->calculateGrid(positionX, positionY, positionZ, objectCount);
    }
    inline unsigned int calculateCellId(float3 positions)
    {
        return static_cast<Derived*>(this)->calculateCellId(positions);
    }
};