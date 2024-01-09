#pragma once
#include "mesh.hpp"
#include <vector>

class sphereMeshGenerator
{
	float radius;
	int parallels;
	int meridians;
	std::vector<Vertex> vertices;
	std::vector<unsigned int> indices;

	void generate();
public:
	sphereMeshGenerator(float radius, int parallels, int meridians);
	InstancedObjectMesh generateMesh();
};