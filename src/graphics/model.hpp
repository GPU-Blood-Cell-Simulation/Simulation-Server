#pragma once

#include <assimp/scene.h>
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
	Model(const char* path);
	Model(Mesh* mesh);

	virtual void draw(const Shader* shader) = 0;

	Mesh* getMesh(unsigned int index);
	unsigned int getVboBuffer(unsigned int index);
	unsigned int getEboBuffer(unsigned int index);
protected:


	std::vector<Texture> textures_loaded;
	std::string directory;
	std::vector<Mesh*> meshes;

	void loadModel(std::string path);
	void processNode(aiNode* node, const aiScene* scene);
	Mesh* processMesh(aiMesh* mesh, const aiScene* scene);

	std::vector<Texture> loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName);
	std::function<Mesh* (std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, std::vector<Texture>&& textures)>createMesh;
};

class SingleObjectModel : public Model
{
public:
	SingleObjectModel(const char* path) : Model(path) {
		createMesh = [&](auto vertices, auto indices, auto textures)->Mesh* {
			return new SingleObjectMesh(std::move(vertices), std::move(indices), std::move(textures));
			};
	}

	SingleObjectModel(Mesh* mesh) : Model(mesh) {
		createMesh = [&](auto vertices, auto indices, auto textures)->Mesh* {
			return new SingleObjectMesh(std::move(vertices), std::move(indices), std::move(textures));
			};
	}

	void draw(const Shader* shader) override;
protected:
};

class InstancedModel : public Model
{
public:
	InstancedModel(const char* path, unsigned int instancesCount);
	InstancedModel(Mesh* mesh, unsigned int instancesCount);

	void draw(const Shader* shader) override;
	unsigned int getCudaOffsetBuffer();

protected:

	unsigned int cudaOffsetBuffer;
	unsigned int instancesCount;
};

class MultipleObjectModel : public Model
{
public:
	MultipleObjectModel() : Model() {}
	MultipleObjectModel(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, std::vector<glm::vec3>& initialPositions, unsigned int objectCount);
	void DuplicateObjects(std::vector<glm::vec3>& initialPositions);

	void draw(const Shader* shader) override;

	unsigned int modelVerticesCount;
	unsigned int objectCount;

protected:
	MultiObjectMesh* mesh;
};

