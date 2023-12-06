file(GLOB_RECURSE CUDA_SOURCES "*.cu")
file(GLOB_RECURSE CUDA_HEADERS "*.cuh")

file(GLOB_RECURSE CPP_SOURCES "*.cpp")
file(GLOB_RECURSE CPP_HEADERS "*.hpp")

add_definitions( -DWINDOW_RENDER)

add_executable(Simulation_server ${CUDA_SOURCES} ${CPP_SOURCES} ${CUDA_HEADERS} ${CPP_HEADERS})

target_link_libraries(
    Simulation_server
    PRIVATE
    mp11
    glad
    GLFW
    KHR
    ${PROJECT_SOURCE_DIR}/Libraries/lib/libglfw3.a
    ${OPENGL_LIBRARIES}
)

target_include_directories(Simulation_server PRIVATE ${OPENGL_INCLUDE_DIRS})

target_compile_options(Simulation_server PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

set_target_properties(Simulation_server PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
