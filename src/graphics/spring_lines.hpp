#pragma once

#include "../meta_factory/blood_cell_factory.hpp"
#include "../graphics/shader.hpp"
#include "../graphics/model.hpp"
#include <array>
#include <memory>
#include <vector>

/// <summary>
/// Represents data of springs in simulation and visualization
/// </summary>
class SpringLines
{
public:
	SpringLines() {};

	/// <summary>
	/// Creates springs data for each blood cell model depending on their VBOs
	/// </summary>
	/// <param name="vbos">Blood cell models VBOs</param>
	void constructSprings(std::array<unsigned int, bloodCellTypeCount>& vbos);

	/// <summary>
	/// Calls OpenGL pipeline of rendering for springs
	/// </summary>
	/// <param name="shader">pointer to shader passed to render pipeline</param>
	void draw(const Shader* shader) const;
private:
	std::array<unsigned int, bloodCellTypeCount> VAOs;
	std::array<std::vector<unsigned int>, bloodCellTypeCount> indexData;
};