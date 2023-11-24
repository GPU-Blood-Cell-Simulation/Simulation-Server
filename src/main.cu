#include <glad/glad.h>
#include "grids/uniform_grid.cuh"
#include "grids/no_grid.cuh"
#include "meta_factory/blood_cell_factory.hpp"
#include "meta_factory/vein_factory.hpp"
#include "objects/blood_cells.cuh"
#include "objects/vein_triangles.cuh"
#include "objects/cylindermesh.hpp"
#include "simulation/simulation_controller.cuh"
#include "utilities/cuda_handle_error.cuh"
#include "graphics/glcontroller.cuh"
#include "objects/cylindermesh.hpp"

#include <curand.h>
#include <curand_kernel.h>
#include <iostream> // for debugging purposes
#include <sstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "config/graphics.hpp"

#ifdef _WIN32 // win only headers

#include "graphics/windowcontroller.hpp"
#define WINDOW_RENDER

#else // server only (linux) headers
#include <unistd.h>
// ...
#endif


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

#ifdef WINDOW_RENDER
    WindowController::GetInstance().ConfigureWindow();
#endif
    // Load GL and set the viewport to match window size
    gladLoadGL();
    glViewport(0, 0, windowWidth, windowHeight);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    VEIN_POLYGON_MODE = GL_FILL;

    // debug
    glEnable(GL_DEBUG_OUTPUT);

    // Main simulation loop
    programLoop();

    // Cleanup
#ifdef WINDOW_RENDER
    glfwTerminate();
#endif
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
    VeinGenerator veinMeshDefinition(cylinderBaseCenter, cylinderHeight, cylinderRadius, cylinderVerticalLayers, cylinderHorizontalLayers);
    SingleObjectMesh veinMesh = veinMeshDefinition.CreateMesh();

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
    graphics::GLController glController(veinMesh, simulationController.initialCellPositions);
    graphics::Camera camera;

#ifdef WINDOW_RENDER
    double lastTime = glfwGetTime();
    WindowController windowController = WindowController::GetInstance();
    windowController.ConfigureInputAndCamera(&camera);

#endif

    // MAIN LOOP HERE - dictated by glfw
    bool shouldBeRunning = true;
    while (shouldBeRunning)
    {
        // Clear 
        glClearColor(1.00f, 0.75f, 0.80f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Calculate particle positions using CUDA
        simulationController.calculateNextFrame();

        // Pass positions to OpenGL
        glController.calculateTriangles(triangles);
        glController.calculatePositions(bloodCells.particles.positions);

        glController.draw(camera);

#ifdef WINDOW_RENDER // graphical render

        glfwSwapBuffers(windowController.window);

        // Show FPS in the title bar
        double currentTime = glfwGetTime();
        double delta = currentTime - lastTime;
        if (delta >= 1.0)
        {
            double fps = double(frameCount) / delta;
            std::stringstream ss;
            ss << "Blood Cell Simulation" << " " << " [" << fps << " FPS]";

            glfwSetWindowTitle(windowController.window, ss.str().c_str());
            lastTime = currentTime;
            frameCount = 0;
        }
        else
        {
            frameCount++;
        }

        // Handle user input
        glfwPollEvents();
        windowController.handleInput();

        shouldBeRunning = !glfwWindowShouldClose(windowController.window);
#else // server calculations

        // Send data to client
            // TODO

        shouldBeRunning = frameCount++ < maxFrames;
#endif
    }
}