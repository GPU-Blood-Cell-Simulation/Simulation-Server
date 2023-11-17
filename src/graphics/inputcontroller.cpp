#include "inputcontroller.hpp"

#include "../config/graphics.hpp"

namespace graphics
{
	// GLFW keyboard callback function
	void InputController::handleUserInput(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		// Get the InputController instance from a custom GLFW pointer
		InputController* controller = static_cast<InputController*>(glfwGetWindowUserPointer(window));

		if (action == GLFW_PRESS)
		{
			// Movement
			switch (key)
			{
			case GLFW_KEY_W:
				controller->pressedKeys.W = true;
				break;
			case GLFW_KEY_S:
				controller->pressedKeys.S = true;
				break;
			case GLFW_KEY_A:
				controller->pressedKeys.A = true;
				break;
			case GLFW_KEY_D:
				controller->pressedKeys.D = true;
				break;
			case GLFW_KEY_SPACE:
				controller->pressedKeys.SPACE = true;
				break;
			case GLFW_KEY_LEFT_SHIFT:
				controller->pressedKeys.SHIFT = true;
				break;
			// Rotation
			case GLFW_KEY_UP:
				controller->pressedKeys.UP = true;
				break;
			case GLFW_KEY_DOWN:
				controller->pressedKeys.DOWN = true;
				break;
			case GLFW_KEY_LEFT:
				controller->pressedKeys.LEFT = true;;
				break;
			case GLFW_KEY_RIGHT:
				controller->pressedKeys.RIGHT = true;
				break;
			// Change view mode
			case GLFW_KEY_V:
				glPolygonMode(GL_FRONT_AND_BACK, (VEIN_POLYGON_MODE = (VEIN_POLYGON_MODE == GL_LINE ? GL_FILL : GL_LINE)));
				break;

			}
		}
		else if (action == GLFW_RELEASE)
		{
			switch (key)
			{
			// Movement
			case GLFW_KEY_W:
				controller->pressedKeys.W = false;
				break;
			case GLFW_KEY_S:
				controller->pressedKeys.S = false;
				break;
			case GLFW_KEY_A:
				controller->pressedKeys.A = false;
				break;
			case GLFW_KEY_D:
				controller->pressedKeys.D = false;
				break;
			case GLFW_KEY_SPACE:
				controller->pressedKeys.SPACE = false;
				break;
			case GLFW_KEY_LEFT_SHIFT:
				controller->pressedKeys.SHIFT = false;
				break;
			// Rotation
			case GLFW_KEY_UP:
				controller->pressedKeys.UP = false;
				break;
			case GLFW_KEY_DOWN:
				controller->pressedKeys.DOWN = false;
				break;
			case GLFW_KEY_LEFT:
				controller->pressedKeys.LEFT = false;
				break;
			case GLFW_KEY_RIGHT:
				controller->pressedKeys.RIGHT = false;
				break;
			}
		}
	}

	// Check pressed keys every frame
	void InputController::adjustParametersUsingInput(Camera& camera)
	{
		if (pressedKeys.W)
			camera.moveForward();
		if (pressedKeys.S)
			camera.moveBack();
		if (pressedKeys.A)
			camera.moveLeft();
		if (pressedKeys.D)
			camera.moveRight();
		if (pressedKeys.SPACE)
			camera.ascend();
		if (pressedKeys.SHIFT)
			camera.descend();
		if (pressedKeys.UP)
			camera.rotateUp();
		if (pressedKeys.DOWN)
			camera.rotateDown();
		if (pressedKeys.LEFT)
			camera.rotateLeft();
		if (pressedKeys.RIGHT)
			camera.rotateRight();
	}
}
