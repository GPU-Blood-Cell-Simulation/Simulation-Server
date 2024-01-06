#pragma once

#include <fstream>
#include <glm/gtc/constants.hpp>
#include <glm/vec3.hpp>
#include <iomanip>
#include <iostream>
#include <tuple>
#include "../graphics/mesh.hpp"
#include <algorithm>
#include <functional>

/// <summary>
/// Generates vein mesh data
/// </summary>
class VeinGenerator
{
	std::vector<Vertex> vertices;
	std::vector<unsigned int> indices;

public:
	VeinGenerator() : indices(veinIndices.data(), veinIndices.data() + veinIndexCount)
	{
		for (auto position : veinPositions)
		{
			auto glmPosition = position.toGLM();
			Vertex v
			{
				glmPosition,
				glm::vec3(glm::normalize(glmPosition - glm::vec3(0, 0, glmPosition.y)))
			};
			vertices.push_back(v);
		}
	}

	/// <summary>
	/// Create vein mesh object
	/// </summary>
	/// <returns>vein mesh</returns>
	SingleObjectMesh CreateMesh()
	{
		return SingleObjectMesh(std::move(vertices), std::move(indices));
	}
};