#pragma once

#include "../objects/particles.cuh"
#include "../utilities/cuda_vec3.cuh"

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
    inline void calculateGrid(const cudaVec3& positions, int objectCount)
    {
        static_cast<Derived*>(this)->calculateGrid(positions);
    }

    /// <summary>
    /// Calculate grid cell id from object position
    /// </summary>
    /// <param name="positions">object position</param>
    /// <returns>cell id</returns>
    inline unsigned int calculateCellId(float3 position)
    {
        return static_cast<Derived*>(this)->calculateCellId(position);
    }
};