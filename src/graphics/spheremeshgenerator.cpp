#include "spheremeshgenerator.hpp"
#include "../meta_factory/blood_cell_factory.hpp"

void sphereMeshGenerator::generate()
{
	const float PI = 3.1415926;

	// top
	vertices.push_back(Vertex{ {0,0,radius},{0,0,1} });

	// poziome
	for (int i = 1; i <= parallels; ++i)
	{
		float n = float(i) / (parallels + 1);
		// index = 
		// pionowe
		for (int j = 0; j <= meridians; ++j)
		{
			float m = float(j) / meridians;

			float x = sin(PI * m) * cos(2 * PI * n);
			float y = sin(PI * m) * sin(2 * PI * n);
			float z = cos(PI * m);

			glm::vec3 normal{ x,y,z };
			vertices.push_back(Vertex{ radius * normal, normal });

			indices.push_back(std::max(0, (i - 2) * meridians + j + 1));
			indices.push_back(std::min(parallels * meridians + 1, (i - 1) * meridians + j + 1));
			indices.push_back(std::min(parallels * meridians + 1, (i - 1) * meridians + (j + 1) % parallels + 1));

			if (i > 0 && i < meridians)
			{
				indices.push_back(std::max(0, (i - 2) * meridians + (j + 1) % parallels + 1));
				indices.push_back(std::max(0, (i - 2) * meridians + j + 1));
				indices.push_back(std::min(parallels * meridians + 1, (i - 1) * meridians + (j + 1) % parallels + 1));
			}
		}
	}

	vertices.push_back(Vertex{ {0,0,-radius},{0,0,-1} });
}

sphereMeshGenerator::sphereMeshGenerator(float radius, int parallels, int meridians) : radius(radius), parallels(parallels), meridians(meridians)
{
	generate();
}

InstancedObjectMesh sphereMeshGenerator::generateMesh()
{
	return InstancedObjectMesh(std::move(vertices), std::move(indices), particleCount);
}
