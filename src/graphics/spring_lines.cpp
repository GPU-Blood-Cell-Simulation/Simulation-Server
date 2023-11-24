#include "spring_lines.hpp"

#include "../meta_factory/blood_cell_factory.hpp"

#include <boost/mp11/algorithm.hpp>
#include <glad/glad.h>
#include "mesh.hpp";


void SpringLines::constructSprings(std::array<unsigned int, bloodCellTypeCount>& vbos)
{
	using IndexList = mp_iota_c<bloodCellTypeCount>;
	mp_for_each<IndexList>([&](auto index)
		{
			int accumulatedParticles = 0;
			using BloodCellDefinition = mp_at_c<BloodCellList, index>;
			// Create index data template (EBO for a particle with given type)
			constexpr int springCount = mp_size<typename BloodCellDefinition::List>::value;

			using IndexListPerBloodCell = mp_iota_c<springCount>;
			mp_for_each<IndexListPerBloodCell>([&](auto j)
				{
					using SpringDefinition = mp_at_c<typename BloodCellDefinition::List, j>;
					indexData[index.value].push_back(accumulatedParticles + SpringDefinition::start);
					indexData[index.value].push_back(accumulatedParticles + SpringDefinition::end);
				});

			// Multiply the index data for other particles of the same type
			int indexDataSize = indexData[index].size();
			for (int j = 1; j < BloodCellDefinition::count; j++)
			{
				for (int k = 0; k < indexDataSize; k++)
				{
					indexData[index.value].push_back(j * BloodCellDefinition::particlesInCell + indexData[index][k]);
				}
			}
			accumulatedParticles += BloodCellDefinition::count * BloodCellDefinition::particlesInCell;

			// setup VAO and EBO (VBO is shared with cuda-mapped position buffer
			glGenVertexArrays(1, &VAOs[index.value]);
			glBindVertexArray(VAOs[index.value]);

			glBindBuffer(GL_ARRAY_BUFFER, vbos[index]);

			// vertex positions
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
			// vertex normals
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
			// vertex texture coords
			glEnableVertexAttribArray(2);
			glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));

			unsigned int EBO;
			glGenBuffers(1, &EBO);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexData[index.value].size() * sizeof(unsigned int), indexData[index.value].data(), GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindVertexArray(0);

		});
}

void SpringLines::draw(const Shader* shader) const
{
	using IndexList = mp_iota_c<bloodCellTypeCount>;
	mp_for_each<IndexList>([&](auto i)
		{
			glBindVertexArray(VAOs[i.value]);
			glDrawElements(GL_LINES, static_cast<GLsizei>(indexData[i.value].size()), GL_UNSIGNED_INT, 0);
			glBindVertexArray(0);
		});
}
