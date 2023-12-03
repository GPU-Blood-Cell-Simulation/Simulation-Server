#include "camera.hpp"

#include "../config/graphics.hpp"
#include "../config/simulation.hpp"

#include <glm/ext/matrix_transform.hpp>

inline constexpr float cameraMovementSpeed = width * cameraMovementSpeedCoefficient;

graphics::Camera::Camera()
{
	calculateView();
}

void graphics::Camera::moveLeft()
{
	position -= right * cameraMovementSpeed;
	calculateView();
}

void graphics::Camera::moveRight()
{
	position += right * cameraMovementSpeed;
	calculateView();
}

void graphics::Camera::moveForward()
{
	position += front * cameraMovementSpeed;
	calculateView();
}

void graphics::Camera::moveBack()
{
	position -= front * cameraMovementSpeed;
	calculateView();
}

void graphics::Camera::ascend()
{
	position += up * cameraMovementSpeed;
	calculateView();
}

void graphics::Camera::descend()
{
	position -= up * cameraMovementSpeed;
	calculateView();
}

void graphics::Camera::rotateLeft()
{
	glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), cameraRotationSpeed, up);
	front = rotation * glm::vec4(front, 1.0f);
	right = rotation * glm::vec4(right, 1.0f);
	calculateView();
}

void graphics::Camera::rotateRight()
{
	glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), -cameraRotationSpeed, up);
	front = rotation * glm::vec4(front, 1.0f);
	right = rotation * glm::vec4(right, 1.0f);
	calculateView();
}

void graphics::Camera::rotateUp()
{
	glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), cameraRotationSpeed, right);
	front = rotation * glm::vec4(front, 1.0f);
	up = rotation * glm::vec4(up, 1.0f);
	calculateView();
}

void graphics::Camera::rotateDown()
{
	glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), -cameraRotationSpeed, right);
	front = rotation * glm::vec4(front, 1.0f);
	up = rotation * glm::vec4(up, 1.0f);
	calculateView();
}

glm::mat4 graphics::Camera::getView() const
{
	return view;
}

glm::vec3 graphics::Camera::getPosition() const
{
	return position;
}

void graphics::Camera::calculateView()
{
	view = glm::lookAt(position, position + front, up);
}
