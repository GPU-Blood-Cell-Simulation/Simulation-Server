#pragma once

#include "../meta_factory/vein_factory.hpp"
#include "../graphics/mesh.hpp"
#include "../utilities/constexpr_vec.hpp"


/// <summary>
/// Generates vein mesh data
/// </summary>
namespace VeinGenerator
{
	/// <summary>
	/// Create vein mesh object
	/// </summary>
	/// <returns>vein mesh</returns>
	SingleObjectMesh createMesh()
	{
		std::vector<unsigned int> indices(veinIndices.data(), veinIndices.data() + veinIndexCount);
		std::vector<Vertex> vertices;	

		for (auto position : veinPositions)
		{
			auto glmPosition = position.toGLM();
			Vertex v
			{
				glmPosition,
				glm::vec3{0,0,0} // Vein normals are currently unnecessary
			};
			vertices.push_back(v);
		}

		return SingleObjectMesh(std::move(vertices), std::move(indices));
	}
};