#pragma once

#include "camera.hpp"

#include <GLFW/glfw3.h>

namespace graphics
{
	// Handles keyboard input and passes it to the camera
	class InputController
	{
	public:
		static void handleUserInput(GLFWwindow* window, int key, int scancode, int action, int mods);
		void adjustParametersUsingInput(Camera& camera);
	private:
		struct PressedKeys
		{
			bool W = false;
			bool S = false;
			bool A = false;
			bool D = false;
			bool SPACE = false;
			bool SHIFT = false;

			bool UP = false;
			bool DOWN = false;
			bool LEFT = false;
			bool RIGHT = false;
		} pressedKeys;
	};
}