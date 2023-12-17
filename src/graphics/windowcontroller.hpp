#pragma once
#include <GLFW/glfw3.h>
#include "camera.hpp"
#include "inputcontroller.hpp"

class WindowController
{
public:
	WindowController();
	~WindowController();

	void ConfigureWindow();
	void ConfigureInputAndCamera(graphics::Camera* camera);

	inline void handleInput()
	{
		if(camera != nullptr)
			inputController.adjustParametersUsingInput(*camera);
	}

	GLFWwindow* window = nullptr;
	graphics::Camera* camera = nullptr;

private:
	graphics::InputController inputController;

};