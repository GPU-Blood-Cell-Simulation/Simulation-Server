add_library(mp11 INTERFACE)
target_include_directories(mp11 INTERFACE include/)

add_library(glad STATIC lib/glad.c)
target_include_directories(glad PUBLIC include/)

add_library(GLFW OBJECT lib/glfw3.lib)
target_include_directories(GLFW INTERFACE include/)

# set(BUILD_STATIC_LIBS ON)
# add_subdirectory(include/glm/)

add_library(glm INTERFACE)
target_include_directories(glm INTERFACE include/)

add_library(KHR INTERFACE)
target_include_directories(KHR INTERFACE include/)
