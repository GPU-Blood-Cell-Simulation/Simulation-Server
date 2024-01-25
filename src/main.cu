#include <glad/glad.h>

#include "config/graphics.hpp"
#include "grids/uniform_grid.cuh"
#include "grids/no_grid.cuh"
#include "meta_factory/blood_cell_factory.hpp"
#include "meta_factory/vein_factory.hpp"
#include "objects/blood_cells.cuh"
#include "objects/vein_triangles.cuh"
#include "simulation/simulation_controller.cuh"
#include "utilities/cuda_handle_error.cuh"
#include "graphics/glcontroller.cuh"
#include "objects/vein_generator.hpp"
#include "objects/sphere_generator.hpp"
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef MULTI_GPU
#include "utilities/nccl_operations.cuh"
#endif


#ifdef WINDOW_RENDER
#   include "graphics/windowcontroller.hpp"
#else

#   include "graphics/offscreencontroller.hpp"
#   include "communication/msg_controller.hpp"

#   undef __noinline__
#   include "graphics/videocontroller.hpp"
#   define __noinline__ __attribute__((noinline))

#endif

#define UNIFORM_TRIANGLES_GRID

//#pragma float_control( except, on )
//// NVIDIA GPU selector for devices with multiple GPUs (e.g. laptops)
//extern "C"
//{
//    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
//}

#ifdef WINDOW_RENDER
#define programLoopFunction void programLoop(WindowController& windowController)
#else
#define programLoopFunction void programLoop(VideoController& streamingController)
#endif

programLoopFunction;

int main()
{
    // Choose main GPU
    CUDACHECK(cudaSetDevice(0));

#ifdef WINDOW_RENDER
    WindowController windowController;
#else
    OffscreenController offscreenController;
    
    VideoController streamingController;
    streamingController.SetUpStreaming(4321, "127.0.0.1");

    if (saveVideoToFile)
        streamingController.SetUpRecording(videoFileName);
    streamingController.StartPlayback();
#endif

    // Load GL and set the viewport to match window size
    gladLoadGL();
    glViewport(0, 0, windowWidth, windowHeight);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    VEIN_POLYGON_MODE = GL_FILL;


    // Main simulation loop
#ifdef WINDOW_RENDER
    programLoop(windowController);
#else
    programLoop(streamingController);
#endif

    // Cleanup
    for (int i = 0; i < gpuCount; i++)
    {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaDeviceReset());
    }
    return 0;
}

// Main simulation loop - upon returning from this function all memory-freeing destructors are called

programLoopFunction
{
    // NCCL
    #ifdef MULTI_GPU
    cudaStream_t streams[gpuCount];
    for (int i = 0; i < gpuCount; i++)
    {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamCreate(&streams[i]));
    }  

    ncclComm_t comms[gpuCount];
    int devs[gpuCount] = { 0, 1, 2, 3}; // GPU_COUNT_DEPENDENT
    NCCLCHECK(ncclCommInitAll(comms, gpuCount, devs));
    cudaSetDevice(0);
    #endif

    int frameCount = 0;
    bool shouldBeRunning = true;
    // Create blood cells
    BloodCells bloodCells;

    // Create vein triangles
    VeinTriangles triangles;
    // Create vein mesh
    SingleObjectMesh veinMesh = VeinGenerator::createMesh();
    InstancedObjectMesh sphereMesh = SphereGenerator::createMesh(10, 10, 1.0f, particleCount);
    // Create grids
    UniformGrid particleGrid(particleGridGpu, particleCount, cellWidth, cellHeight, cellDepth);
#ifdef UNIFORM_TRIANGLES_GRID
    UniformGrid triangleCentersGrid(veinGridGpu, triangleCount, cellWidthTriangles, cellHeightTriangles, cellDepthTriangles);
#else
    NoGrid triangleCentersGrid;
#endif

    // Create the main simulation controller and inject its dependencies
    sim::SimulationController simulationController(bloodCells, triangles, &particleGrid, &triangleCentersGrid);
    
    // Create a graphics controller
    graphics::GLController glController(veinMesh, sphereMesh, simulationController);
    graphics::Camera camera;
#ifdef WINDOW_RENDER
    double lastTime = glfwGetTime();
    windowController.ConfigureInputAndCamera(&camera);
#else
    MsgController msgController(4322);
    msgController.setCamera(&camera);
    msgController.setStreamEndCallback([&shouldBeRunning]() { shouldBeRunning = false; });
#endif

    std::cout << "started rendering...\n";
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // MAIN LOOP HERE - dictated by glfw
    while (shouldBeRunning)
    {
        // Calculate grids
#ifdef MULTI_GPU
        bloodCells.broadcastParticles(comms, streams);
        triangles.broadcastPositionsAndVelocities(comms, streams);
        std::thread g1thread([&](){
            particleGrid.calculateGrid(bloodCells.particles.positions[particleGridGpu], particleCount);
        });
        std::thread g2thread([&](){
            // Recalculate triangles centers
            static CudaThreads threads(triangleCount);
            triangles.calculateCenters(threads.blocks, threads.threadsPerBlock);
            // Calculate grid
            triangleCentersGrid.calculateGrid(triangles.centers, triangleCount);
        });
#else
        particleGrid.calculateGrid(bloodCells.particles.positions[particleGridGpu], particleCount);
        triangleCentersGrid.calculateGrid(triangles.centers, triangleCount);
#endif        
        // Clear 
        glClearColor(1.00f, 0.75f, 0.80f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Pass positions to OpenGL
        glController.calculateTriangles(triangles);
        glController.calculatePositions(bloodCells.particles.positions[0]);

        glController.draw(camera);

#ifdef MULTI_GPU
        // Join grid threads
        g1thread.join();
        g2thread.join();

        // Broadcast grid data
        particleGrid.broadcastGrid(comms, streams);
        triangleCentersGrid.broadcastGrid(comms, streams);
#endif

        // Calculate particle positions using CUDA
        simulationController.calculateNextFrame();

#ifdef MULTI_GPU
        // Multi gpu reduction - merge forces calculated on separate devices
        bloodCells.reduceForces(comms, streams);
        triangles.reduceForces(comms, streams);

#endif
        // Propagate forces into velocities and positions
        simulationController.propagateAll();     

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
        streamingController.SendFrame();
        msgController.handleMsgs();
#endif
        //std::cout << frameCount << "\n";
        if (++frameCount > maxFrames) {
            shouldBeRunning = false;
        }
    }
    
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double seconds = (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0;
    std::cout << "finished rendering in " << seconds << " seconds\n";
    std::cout << "Average framerate: " << (frameCount - 1) / seconds << " fps\n";
    std::cout << "Average single frame render time: " << seconds / (frameCount - 1) << " fps\n";
#ifndef WINDOW_RENDER
    msgController.successfulStreamEndInform();
#endif
        
  
#ifdef MULTI_GPU
    for(int i = 0; i < gpuCount; i++)
    {
        ncclCommDestroy(comms[i]);
        cudaStreamDestroy(streams[i]);
    }
#endif
}