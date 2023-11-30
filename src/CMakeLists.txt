file(GLOB_RECURSE CUDA_SOURCES "*.cu")
file(GLOB_RECURSE CUDA_HEADERS "*.cuh")

file(GLOB_RECURSE CPP_SOURCES "*.cpp")
file(GLOB_RECURSE CPP_HEADERS "*.hpp")

include(CMakePrintHelpers)
cmake_print_variables(CPP_SOURCES)


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
    ${PROJECT_SOURCE_DIR}/Libraries/lib/libglfw3.a
)

# target_compile_features(Simulation_server PUBLIC cxx_std_17)

# set_target_properties(Simulation_server PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(Simulation_server PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#target_compile_options(Simulation_server PRIVATE --expt-relaxed-constexpr)
