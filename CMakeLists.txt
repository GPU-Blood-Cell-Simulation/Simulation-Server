cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(SimulationServer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_subdirectory(Libraries/)
add_subdirectory(src/)
