file(GLOB_RECURSE CUDA_SOURCES "*.cu")
file(GLOB_RECURSE CUDA_HEADERS "*.cuh")

file(GLOB_RECURSE CPP_SOURCES "*.cpp")
file(GLOB_RECURSE CPP_SOURCES "*.hpp")

include(CMakePrintHelpers)
cmake_print_variables(CUDA_SOURCES)



# add_library(srcs STATIC
#     ${CUDA_HEADERS}
#     ${CPP_HEADERS}
#     ${CUDA_SOURCES}
#     ${CPP_SOURCES}
# )

add_executable(Simulation_server ${CUDA_SOURCES} ${CPP_SOURCES} ${CUDA_HEADERS} ${CPP_HEADERS})

# include_directories(${PROJECT_SOURCE_DIR}src/)

target_link_libraries(
    Simulation_server
    PRIVATE
    mp11
    glad
    GLFW
    glm
    KHR
)

# target_compile_features(Simulation_server PUBLIC cxx_std_17)

# set_target_properties(Simulation_server PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(Simulation_server PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(Simulation_server PRIVATE --expt-relaxed-constexpr)
