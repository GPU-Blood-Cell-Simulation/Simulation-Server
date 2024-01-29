#pragma once
#include <GLFW/glfw3.h>
#include "camera.hpp"
#include "inputcontroller.hpp"
#include <memory>

/// <summary>
/// Controlls visualization main window
/// </summary>
class WindowController
{
	graphics::InputController inputController;

public:
	WindowController();
	~WindowController();

	GLFWwindow* window = nullptr;
	graphics::Camera* camera;

	void ConfigureInputAndCamera(graphics::Camera* camera);

	inline void handleInput()
	{
		if(camera != nullptr)
			inputController.adjustParametersUsingInput(*camera);
	}
};