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
	graphics::Camera* camera = nullptr;

	static WindowController& GetInstance();
	void ConfigureWindow();
	void ConfigureInputAndCamera(graphics::Camera*);

	inline void handleInput()
	{
		if(camera != nullptr)
			inputController.adjustParametersUsingInput(*camera);
	}
};