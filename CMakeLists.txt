cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(SimulationServer LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

IF(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 50)
endif()

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)
find_package(PkgConfig REQUIRED)
pkg_search_module(gstreamer REQUIRED IMPORTED_TARGET gstreamer-1.0>=1.4)
pkg_search_module(gstreamer-app REQUIRED IMPORTED_TARGET gstreamer-app-1.0>=1.4)
find_package(X11 REQUIRED)

add_definitions( -DWINDOW_RENDER)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")

# Set CUDA architecture (adjust based on your GPU)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_50)

#add_definitions(${OPENGL_DEFINITIONS})

# note that the order is important for setting the libs

set (CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl")
set (CMAKE_CUDA_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl")

add_library(GLAD Libraries/lib/glad.c)
set(LIBS ${LIBS} GLAD glfw3)

include_directories(${CMAKE_SOURCE_DIR}/Libraries/include/)
link_directories(${CMAKE_SOURCE_DIR}/Libraries/lib/)

file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
file(GLOB_RECURSE CUDA_HEADERS "src/*.cuh")

file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
file(GLOB_RECURSE CPP_HEADERS "src/*.hpp" "src/*.h")

# Create an executable target
add_executable(Simulation_server ${CUDA_SOURCES} ${CPP_SOURCES} ${CUDA_HEADERS} ${CPP_HEADERS})

target_compile_options(Simulation_server PRIVATE -ggdb3 $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(Simulation_server PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(Simulation_server
  ${LIBS}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  ${CUDA_curand_LIBRARY}
  OpenGL::OpenGL OpenGL::EGL
  PkgConfig::gstreamer
  PkgConfig::gstreamer-app
)

set(LINKER_FLAGS  "${LINKER_FLAGS} -lm")

set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${COMPILE_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${LINKER_FLAGS}")
