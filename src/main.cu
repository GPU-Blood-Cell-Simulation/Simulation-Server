#include "grids/uniform_grid.cuh"
#include "grids/no_grid.cuh"
#include "meta_factory/blood_cell_factory.hpp"
#include "meta_factory/vein_factory.hpp"
#include "objects/blood_cells.cuh"
#include "objects/vein_triangles.cuh"
#include "objects/cylindermesh.hpp"
#include "simulation/simulation_controller.cuh"
#include "utilities/cuda_handle_error.cuh"

#include <curand.h>
#include <curand_kernel.h>
#include <iostream> // for debugging purposes
#include <sstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define UNIFORM_TRIANGLES_GRID

//#pragma float_control( except, on )
//// NVIDIA GPU selector for devices with multiple GPUs (e.g. laptops)
//extern "C"
//{
//    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
//}

void programLoop();

int main()
{
    // Choose which GPU to run on, change this on a multi-GPU system.
    HANDLE_ERROR(cudaSetDevice(0));
    
    // Main simulation loop
    
    programLoop();

    // Cleanup

    HANDLE_ERROR(cudaDeviceReset());

    return 0;
}

// Main simulation loop - upon returning from this function all memory-freeing destructors are called
void programLoop()
{
    int frameCount = 0;

    // Create blood cells
    BloodCells bloodCells;

    // Create vein mesh
    VeinGenerator veinGenerator(cylinderBaseCenter, cylinderHeight, cylinderRadius, cylinderVerticalLayers, cylinderHorizontalLayers);

    // Create vein triangles
    VeinTriangles triangles(veinGenerator.getVertices(), veinGenerator.getIndices(), veinGenerator.getSpringLengths());

    // Create grids
    UniformGrid particleGrid(particleCount, 20, 20, 20);
#ifdef UNIFORM_TRIANGLES_GRID
    UniformGrid triangleCentersGrid(triangles.triangleCount, 10, 10, 10);
#else
    NoGrid triangleCentersGrid;
#endif

    // Create the main simulation controller and inject its dependencies
    sim::SimulationController simulationController(bloodCells, triangles, &particleGrid, &triangleCentersGrid);

    // Create a graphics controller

    // MAIN LOOP HERE - dictated by glfw

    while (frameCount++ < maxFrames)
    {
        // Calculate particle positions using CUDA
        simulationController.calculateNextFrame();

        // Send data to client
            // TODO

    }
}