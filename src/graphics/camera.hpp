#pragma once

#include "../config/simulation.hpp"
#include "../meta_factory/vein_factory.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>


namespace graphics
{
	/// <summary>
	/// Represents camera vision for visualization purpose
	/// </summary>
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

		/// <summary>
		/// Get's view matrix
		/// </summary>
		/// <returns>view matrix</returns>
		const glm::mat4& getView() const;

		/// <summary>
		/// Get's actual camera position
		/// </summary>
		/// <returns>vector camera of position</returns>
		const glm::vec3& getPosition() const;
	private:
		glm::mat4 view;

		glm::vec3 position {(minX + maxX) / 2, (minY + maxY) / 2, 20 * maxZ};
		glm::vec3 front {0, 0, -1};
		glm::vec3 up {0, 1, 0};
		glm::vec3 right = glm::cross(front, up);
		

		void calculateView();
	};
}
