cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(SimulationServer LANGUAGES C CXX CUDA)

option(WINDOW_RENDERING "Switches between window rendering and streaming mode" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

IF(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 50)
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_50)

#
# Adding compilation flags
#
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")

set (CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl -lm")
set (CMAKE_CUDA_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl -lm")

#
# Program sources
#
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
file(GLOB_RECURSE CUDA_HEADERS "src/*.cuh")
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
file(GLOB_RECURSE CPP_HEADERS "src/*.hpp" "src/*.h")

#
# Common packages
#
find_package(CUDA REQUIRED)
add_library(GLAD Libraries/lib/glad.c)

set(LIBS
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  ${CUDA_curand_LIBRARY}
  GLAD
  ${CMAKE_SOURCE_DIR}/Libraries/lib/libenet.a
)

#
# Program mode specific options
#
if(WINDOW_RENDERING)
  #
  # Window rendering
  #
  find_package(OpenGL REQUIRED COMPONENTS OpenGL)
  find_package(X11 REQUIRED)
  add_definitions( -DWINDOW_RENDER)

  set(LIBS
    ${LIBS}
    OpenGL::OpenGL
    ${CMAKE_SOURCE_DIR}/Libraries/lib/libglfw3.a
  )

  list(REMOVE_ITEM CPP_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/graphics/offscreencontroller.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/graphics/streamingcontroller.cpp"
  )

  list(REMOVE_ITEM CPP_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/graphics/offscreencontroller.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/graphics/streamingcontroller.hpp"
  )

else()
  #
  # Streaming
  #
  find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)
  find_package(PkgConfig REQUIRED)
  pkg_search_module(gstreamer REQUIRED IMPORTED_TARGET gstreamer-1.0>=1.4)
  pkg_search_module(gstreamer-app REQUIRED IMPORTED_TARGET gstreamer-app-1.0>=1.4)

  set(LIBS
    ${LIBS}
    OpenGL::OpenGL OpenGL::EGL
    PkgConfig::gstreamer
    PkgConfig::gstreamer-app
  )

  list(REMOVE_ITEM CPP_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/graphics/windowcontroller.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/graphics/inputcontroller.cpp"
  )
  list(REMOVE_ITEM CPP_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/graphics/windowcontroller.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/graphics/inputcontroller.hpp"
  )

endif()

include_directories(${CMAKE_SOURCE_DIR}/Libraries/include/)
link_directories(${CMAKE_SOURCE_DIR}/Libraries/lib/)

#
# Defining executable target
#
add_executable(Simulation_server ${CUDA_SOURCES} ${CPP_SOURCES} ${CUDA_HEADERS} ${CPP_HEADERS})

target_compile_options(Simulation_server PRIVATE -ggdb3 $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --maxrregcount=40>)
set_target_properties(Simulation_server PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(Simulation_server ${LIBS})

#
# Copy directory with shaders
#
file(COPY Shaders DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

#
# Tests
#
add_subdirectory("${PROJECT_SOURCE_DIR}/Libraries/googletest-1.14.0/")
add_subdirectory("tests/")
