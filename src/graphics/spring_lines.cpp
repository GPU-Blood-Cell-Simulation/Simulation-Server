#include "spring_lines.hpp"

#include "../meta_factory/blood_cell_factory.hpp"

#include <boost/mp11/algorithm.hpp>
#include <glad/glad.h>

SpringLines::SpringLines(unsigned int VBO)
{
	int accumulatedParticles = 0;
	using IndexList = mp_iota_c<bloodCellTypeCount>;
	mp_for_each<IndexList>([&](auto i)
	{
		using BloodCellDefinition = mp_at_c<BloodCellList, i>;
		// Create index data template (EBO for a particle with given type)
		constexpr int springCount = mp_size<typename BloodCellDefinition::List>::value;

		using IndexListPerBloodCell = mp_iota_c<springCount>;
		mp_for_each<IndexListPerBloodCell>([&](auto j)
		{
			using SpringDefinition = mp_at_c<typename BloodCellDefinition::List, j>;
			indexData[i].push_back(accumulatedParticles + SpringDefinition::start);
			indexData[i].push_back(accumulatedParticles + SpringDefinition::end);
		});

		// Multiply the index data for other particles of the same type
		int indexDataSize = indexData[i].size();
		for (int j = 1; j < BloodCellDefinition::count; j++)
		{	
			for (int k = 0; k < indexDataSize; k++)
			{
				indexData[i].push_back(j * BloodCellDefinition::particlesInCell + indexData[i][k]);
			}
		}
		accumulatedParticles += BloodCellDefinition::count * BloodCellDefinition::particlesInCell;

		// setup VAO and EBO (VBO is shared with cuda-mapped position buffer
		glGenVertexArrays(1, &VAOs[i]);
		glBindVertexArray(VAOs[i]);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), NULL);
		unsigned int EBO;
		glGenBuffers(1, &EBO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexData[i].size() * sizeof(unsigned int), indexData[i].data(), GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);

	});

}

void SpringLines::draw(const Shader* shader) const
{
	using IndexList = mp_iota_c<bloodCellTypeCount>;
	mp_for_each<IndexList>([&](auto i)
	{
		glBindVertexArray(VAOs[i]);
		glDrawElements(GL_LINES, static_cast<GLsizei>(indexData[i].size()), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
	});

	
}
