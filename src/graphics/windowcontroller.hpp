#pragma once
#include <GLFW/glfw3.h>
#include "camera.hpp"
#include "inputcontroller.hpp"
#include <memory>

class WindowController
{
	graphics::InputController inputController;

public:
	WindowController();
	~WindowController();

	GLFWwindow* window = nullptr;
	graphics::Camera* camera;

	WindowController();
	~WindowController();

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