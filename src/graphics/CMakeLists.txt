# add_library(camera camera.cpp camera.hpp)
# target_link_libraries(camera PRUBLIC config)

# add_library(input_controller glcontroller.cuh glcontroller.cu)
# target_link_libraries(input_controller PRUBLIC config)

add_library(glController
    camera.hpp
    camera.cpp
    light.hpp
    inputcontroller.hpp
    inputcontroller.cpp
    mesh.hpp
    mesh.cpp
    shader.hpp
    spring_lines.hpp
    spring_lines.cpp
)

target_link_libraries(glController PRIVATE glm)
target_link_libraries(glController PRIVATE glad)
target_link_libraries(glController PRIVATE mp11)
