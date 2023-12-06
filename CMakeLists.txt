cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(SimulationServer LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)
find_package(glfw3 3.3 REQUIRED)

set(glm_DIR /usr/lib/cmake/glm) # if necessary
find_package(glm REQUIRED)

# Set CUDA architecture (adjust based on your GPU)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_50)

# Include directories
include_directories(${GLFW_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS} ${GLM_INCLUDE_DIRS})

# Add CUDA include directories
include_directories(${CUDA_INCLUDE_DIRS})

# Add GLAD library
add_library(GLAD "Libraries/lib/glad.c")
include_directories("Libraries/include")

file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
file(GLOB_RECURSE CUDA_HEADERS "src/*.cuh")

file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
file(GLOB_RECURSE CPP_HEADERS "src/*.hpp")

add_definitions( -DWINDOW_RENDER)

add_definitions(${OPENGL_DEFINITIONS} ${GLFW_DEFINITIONS} ${GLM_DEFINITIONS})
add_compile_options(-Wall -Wextra)


SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -lrt -lm -ldl")

# Add your source files
set(SOURCE_FILES
    ${CUDA_SOURCES} ${CPP_SOURCES} ${CUDA_HEADERS} ${CPP_HEADERS}
)

# Create an executable target
add_executable(Simulation_server ${SOURCE_FILES})
target_compile_options(Simulation_server PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# Link libraries
target_link_libraries(Simulation_server PRIVATE ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLM_LIBRARIES} ${CUDA_LIBRARIES} GLAD
${PROJECT_SOURCE_DIR}/Libraries/lib/libglfw3.a)

# Set properties for CUDA files
set_target_properties(Simulation_server PROPERTIES CUDA_SEPARABLE_COMPILATION ON)