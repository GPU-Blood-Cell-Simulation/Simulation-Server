#pragma once

#include "../meta_factory/blood_cell_factory.hpp"
#include "../graphics/shader.hpp"

#include <array>
#include <memory>
#include <vector>

class SpringLines
{
public:
	SpringLines(unsigned int VBO);
	void draw(const Shader* shader) const;
private:
	std::array<unsigned int, bloodCellTypeCount> VAOs;
	std::array<std::vector<unsigned int>, bloodCellTypeCount> indexData;
};