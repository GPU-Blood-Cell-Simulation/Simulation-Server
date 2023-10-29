#pragma once

#include <fstream>
#include <glm/gtc/constants.hpp>
#include <glm/vec3.hpp>
#include <iomanip>
#include <iostream>
#include <tuple>

class VeinGenerator
{
	std::vector<glm::vec3> vertices;
	std::vector<unsigned int> indices;

	const float radius, height;
	const unsigned int vLayers, hLayers;
	glm::vec3 basisOrigin;

	std::tuple<float, float, float> springLengths;

public:
	VeinGenerator(glm::vec3 basisOrigin, float height, float radius,
		unsigned int cylinderVerticalLayers, unsigned int cylinderHorizontalLayers) : basisOrigin(basisOrigin),
		radius(radius), height(height), vLayers(cylinderVerticalLayers), hLayers(cylinderHorizontalLayers)
	{
		float triangleH = height / vLayers;
		float radianBatch = 2 * glm::pi<float>() / hLayers;
		float triangleBase = radianBatch * radius;
		for (unsigned int i = 0; i < vLayers; ++i)
		{
			float h = i * triangleH /*- height / 2*/;
			for (unsigned int j = 0; j < hLayers; ++j)
			{
				vertices.emplace_back(
					glm::vec3(radius * cos(j * radianBatch),
					h, radius * sin(j * radianBatch)) + basisOrigin
				);
				if (i < vLayers - 1)
				{
					int nextj = (j + 1) % hLayers;
					indices.push_back((i + 1) * hLayers + j);
					indices.push_back(i * hLayers + j);
					indices.push_back(i * hLayers + nextj);
					indices.push_back((i + 1) * hLayers + j);
					indices.push_back(i * hLayers + nextj);
					indices.push_back((i + 1) * hLayers + nextj);
				}
			}
		}
		// Vertex distance debug info
		/*
		std::cout << "Horizontal (j): " << length(vertices[0].Position - vertices[1].Position) << std::endl;
		std::cout << "Vertical (i): " << length(vertices[0].Position - vertices[hLayers].Position) << std::endl;
		std::cout << "Diagonal: " << length(vertices[0].Position - vertices[hLayers + 1].Position) << std::endl;
		exit(0);*/

		springLengths = std::make_tuple(
			length(vertices[0] - vertices[1]),
			length(vertices[0] - vertices[hLayers]),
			length(vertices[0] - vertices[hLayers + 1])
		);
	}

	inline const std::vector<glm::vec3>& getVertices() const
	{
		return vertices;
	}

	inline const std::vector<unsigned int>& getIndices() const
	{
		return indices;
	}

	inline std::tuple<float, float, float> getSpringLengths() const
	{
		return springLengths;
	}
};