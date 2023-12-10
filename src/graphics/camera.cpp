#include "camera.hpp"

#include "../config/graphics.hpp"
#include "../config/simulation.hpp"

#include <glm/gtc/matrix_transform.hpp>

namespace graphics
{
	inline constexpr float cameraMovementSpeed = width * cameraMovementSpeedCoefficient;

	Camera::Camera()
	{
		calculateView();
	}

	void Camera::moveLeft()
	{
		position -= right * cameraMovementSpeed;
		calculateView();
	}

	void Camera::moveRight()
	{
		position += right * cameraMovementSpeed;
		calculateView();
	}

	void Camera::moveForward()
	{
		// position += front * cameraMovementSpeed;
		position.x += front.x * cameraMovementSpeed;
		position.y += front.y * cameraMovementSpeed;
		position.z += front.z * cameraMovementSpeed;
		calculateView();
	}

	void Camera::moveBack()
	{
		position -= front * cameraMovementSpeed;
		calculateView();
	}

	void Camera::ascend()
	{
		position += up * cameraMovementSpeed;
		calculateView();
	}

	void Camera::descend()
	{
		position -= up * cameraMovementSpeed;
		calculateView();
	}

	void Camera::rotateLeft()
	{
		glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), cameraRotationSpeed, up);
		front = rotation * glm::vec4(front, 1.0f);
		right = rotation * glm::vec4(right, 1.0f);
		calculateView();
	}

	void Camera::rotateRight()
	{
		glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), -cameraRotationSpeed, up);
		front = rotation * glm::vec4(front, 1.0f);
		right = rotation * glm::vec4(right, 1.0f);
		calculateView();
	}

	void Camera::rotateUp()
	{
		glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), cameraRotationSpeed, right);
		front = rotation * glm::vec4(front, 1.0f);
		up = rotation * glm::vec4(up, 1.0f);
		calculateView();
	}

	void Camera::rotateDown()
	{
		glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), -cameraRotationSpeed, right);
		front = rotation * glm::vec4(front, 1.0f);
		up = rotation * glm::vec4(up, 1.0f);
		calculateView();
	}

	const glm::mat4& Camera::getView() const
	{
		return view;
	}

	const glm::vec3& Camera::getPosition() const
	{
		return position;
	}

	void Camera::calculateView()
	{
		view = glm::lookAt(position, position + front, up);
	}
}