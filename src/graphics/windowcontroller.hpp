#pragma once
#include <GLFW/glfw3.h>
#include "camera.hpp"
#include "inputcontroller.hpp"

class WindowController
{
	WindowController();
	graphics::InputController inputController;

public:

	GLFWwindow* window = nullptr;
	graphics::Camera camera;
	
	static WindowController& GetInstance();
	void ConfigureWindow();
	void ConfigureInput();

	inline void handleInput()
	{
		inputController.adjustParametersUsingInput(camera);
	}
};