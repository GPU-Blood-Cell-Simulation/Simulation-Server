#pragma once

#include "../config/simulation.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>


namespace graphics
{
	class Camera
	{
	public:
		Camera();

		void moveLeft();
		void moveRight();
		void moveForward();
		void moveBack();
		void ascend();
		void descend();

		void rotateLeft();
		void rotateRight();
		void rotateUp();
		void rotateDown();

		glm::mat4 getView() const;
		glm::vec3 getPosition() const;
	private:
		glm::mat4 view;

		glm::vec3 position = glm::vec3(width / 2, height / 2, depth * 3);
		glm::vec3 front = glm::vec3(0, 0, -1);
		glm::vec3 up = glm::vec3(0, 1, 0);
		glm::vec3 right = glm::cross(front, up);
		

		void calculateView();
	};
}
