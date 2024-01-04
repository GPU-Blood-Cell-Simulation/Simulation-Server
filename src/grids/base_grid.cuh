#pragma once

#include "../objects/particles.cuh"

/// <summary>
/// Base grid class - uses the Curiously Recurring Template Pattern to provide compile-time inheritance. Use it with std::variant and std::visit
/// </summary>
/// <typeparam name="Derived">A concrete grid implementation</typeparam>
template<typename Derived>
class BaseGrid
{
protected:
    BaseGrid() {}
public:

    /// <summary>
    /// Recalculate grid basing on particles positions
    /// </summary>
    /// <param name="particles">simulation particles</param>
    inline void calculateGrid(const Particles& particles)
    {
        static_cast<Derived*>(this)->calculateGrid(particles);
    }

    /// <summary>
    /// Recalculate grid basing on objects positions
    /// </summary>
    /// <param name="positionX">device buffer of X's of positions</param>
    /// <param name="positionY">device buffer of Y's of positions</param>
    /// <param name="positionZ">device buffer of Z's of positions</param>
	/// <param name="objectCount">number of objects</param>
    inline void calculateGrid(const float* positionX, const float* positionY, const float* positionZ, int objectCount)
    {
        static_cast<Derived*>(this)->calculateGrid(positionX, positionY, positionZ, objectCount);
    }

    /// <summary>
    /// Calculate grid cell id from object position
    /// </summary>
    /// <param name="positions">object position</param>
    /// <returns>cell id</returns>
    inline unsigned int calculateCellId(float3 positions)
    {
        return static_cast<Derived*>(this)->calculateCellId(positions);
    }
};