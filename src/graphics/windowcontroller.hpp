#pragma once
#include <GLFW/glfw3.h>
#include "camera.hpp"
#include "inputcontroller.hpp"
#include <memory>

class WindowController
{
	WindowController();
	graphics::InputController inputController;
	~WindowController();
public:

	GLFWwindow* window = nullptr;
	graphics::Camera* camera;

	static WindowController& GetInstance();
	void ConfigureWindow();
	void ConfigureInputAndCamera(graphics::Camera* camera);

	inline void handleInput()
	{
		if(camera != nullptr)
			inputController.adjustParametersUsingInput(*camera);
	}
};