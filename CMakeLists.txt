cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(SimulationServer LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_CUDA_STANDARD_REQUIRED ON)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 50)
endif()

# Find required packages
find_package(CUDA REQUIRED)
find_package(GLM REQUIRED)
message(STATUS "GLM included at ${GLM_INCLUDE_DIR}")
find_package(GLFW3 REQUIRED)
message(STATUS "Found GLFW3 in ${GLFW3_INCLUDE_DIR}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")

# Set CUDA architecture (adjust based on your GPU)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_50)

find_package(OpenGL REQUIRED)
add_definitions(${OPENGL_DEFINITIONS})
find_package(X11 REQUIRED)
# note that the order is important for setting the libs
# use pkg-config --libs $(pkg-config --print-requires --print-requires-private glfw3) in a terminal to confirm
set(LIBS ${GLFW3_LIBRARY} X11 Xrandr Xinerama Xi Xxf86vm Xcursor GL dl pthread freetype)
set (CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl")
set (CMAKE_CUDA_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl")


add_library(GLAD Libraries/lib/glad.c)
set(LIBS ${LIBS} GLAD)

include_directories(${CMAKE_SOURCE_DIR}/Libraries/include/)

file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
file(GLOB_RECURSE CUDA_HEADERS "src/*.cuh")

file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
file(GLOB_RECURSE CPP_HEADERS "src/*.hpp" "src/*.h")

add_definitions( -DWINDOW_RENDER)

# Create an executable target
add_executable(Simulation_server ${CUDA_SOURCES} ${CPP_SOURCES} ${CUDA_HEADERS} ${CPP_HEADERS})
target_link_libraries(Simulation_server ${LIBS}
${CUDA_LIBRARIES}
${CUDA_CUDA_LIBRARY}
${CUDA_curand_LIBRARY}
)

target_compile_options(Simulation_server PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

set_target_properties(Simulation_server PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(LINKER_FLAGS  "${LINKER_FLAGS} -lm")

set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${COMPILE_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${LINKER_FLAGS}")