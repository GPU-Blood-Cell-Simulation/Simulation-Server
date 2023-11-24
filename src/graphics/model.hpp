#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "mesh.hpp"
#include "shader.hpp"


class Model
{
public:
	Model() {};

	void draw(const Shader* shader) const;

	unsigned int getVboBuffer(unsigned int index);
	unsigned int getEboBuffer(unsigned int index);

	void addMesh(Mesh& mesh);

protected:

	std::string directory;
	std::vector<Mesh> meshes;
};


class InstancedModel : public Model
{
public:
	InstancedModel(Mesh& mesh, unsigned int instancesCount);
	unsigned int getCudaOffsetBuffer();

protected:

	unsigned int cudaOffsetBuffer;
	unsigned int instancesCount;
};

class MultipleObjectModel : public Model
{
public:
	MultipleObjectModel() : Model() { verticesInitialCount = objectCount = 0; }
	MultipleObjectModel(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, std::vector<glm::vec3>& initialPositions, unsigned int objectCount);

protected:
	unsigned int verticesInitialCount;
	unsigned int objectCount;
};

