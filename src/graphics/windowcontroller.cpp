#include "windowcontroller.hpp"
#include "../config/graphics.hpp"

WindowController::WindowController() { }

WindowController::~WindowController()
{
    glfwDestroyWindow(window);
    glfwTerminate();
}

WindowController& WindowController::GetInstance()
{
    static WindowController instance;
	return instance;
}

void WindowController::ConfigureWindow()
{
    if (!glfwInit())
        exit(-1);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create a windowed mode window and its OpenGL context
    this->window = glfwCreateWindow(windowWidth, windowHeight, "Blood Cell Simulation", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(-1);
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);
}

void WindowController::ConfigureInputAndCamera(graphics::Camera* _camera)
{
    this->camera = _camera;
    // Set up GLFW to work with inputController
    glfwSetWindowUserPointer(window, &inputController);
    glfwSetKeyCallback(window, inputController.handleUserInput);
}
