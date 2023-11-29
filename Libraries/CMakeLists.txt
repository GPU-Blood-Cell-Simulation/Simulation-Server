add_library(mp11 INTERFACE)
target_include_directories(mp11 INTERFACE include/)

add_library(glad INTERFACE)
target_include_directories(glad INTERFACE include/)

add_library(GLFW OBJECT lib/glfw3.lib)
target_include_directories(GLFW INTERFACE include/)

add_library(glm INTERFACE)
target_include_directories(glm INTERFACE include/)

add_library(KHR INTERFACE)
target_include_directories(KHR INTERFACE include/)
