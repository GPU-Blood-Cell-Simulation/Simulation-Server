#pragma once

#include "../meta_factory/blood_cell_factory.hpp"
#include "../graphics/shader.hpp"
#include "../graphics/model.hpp"
#include <array>
#include <memory>
#include <vector>

class SpringLines
{
public:
	SpringLines() {};
	void constructSprings(MultipleObjectModel(*bloodCellModels)[bloodCellTypeCount]);
	void draw(const Shader* shader) const;
private:
	std::array<unsigned int, bloodCellTypeCount> VAOs;
	std::array<std::vector<unsigned int>, bloodCellTypeCount> indexData;
};