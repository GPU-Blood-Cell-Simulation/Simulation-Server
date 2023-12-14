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

// TODO: remove
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
		// Vertex distance debug info
		
		/*std::cout << "Horizontal (j): " << length(vertices[0].Position - vertices[1].Position) << std::endl;
		std::cout << "Vertical (i): " << length(vertices[0].Position - vertices[hLayers].Position) << std::endl;
		std::cout << "Diagonal: " << length(vertices[0].Position - vertices[hLayers + 1].Position) << std::endl;
		exit(0);

		springLengths = std::make_tuple(
			length(vertices[0].position - vertices[1].position),
			length(vertices[0].position - vertices[hLayers].position),
			length(vertices[0].position - vertices[hLayers + 1].position)
		);*/
	}

	inline const std::vector<glm::vec3> getVertices() const
	{
		std::vector<glm::vec3> positions(vertices.size());
		std::transform(vertices.cbegin(), vertices.cend(), positions.begin(), [&](auto& v) {
			return v.position;
		});
		return positions;
	}

	inline const std::vector<unsigned int>& getIndices() const
	{
		return indices;
	}

	inline std::tuple<float, float, float> getSpringLengths() const
	{
		//return springLengths;
		return std::tuple(0,0,0);
	}

	SingleObjectMesh CreateMesh()
	{
		return SingleObjectMesh(std::move(vertices), std::move(indices));
	}
};