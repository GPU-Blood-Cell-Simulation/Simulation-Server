cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(SimulationServer LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_subdirectory(Libraries/)
add_subdirectory(src/)

# SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}" )
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
