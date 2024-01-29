#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "mesh.hpp"
#include "shader.hpp"


/// <summary>
/// Abstract model class
/// </summary>
class Model
{
public:

	/// <summary>
	/// Calls OpenGL render pipeline for underlying meshes
	/// </summary>
	/// <param name="shader">pointer to shader passed to the pipeline</param>
	virtual void draw(const Shader* shader) const = 0;

	/// <summary>
	/// Gets Vertex Buffer Object of certain mesh from model
	/// </summary>
	/// <param name="index">index of mesh</param>
	/// <returns>Vertex Buffer Object</returns>
	virtual unsigned int getVboBuffer(unsigned int index) = 0;
	
	/// <summary>
	/// Gets Element Buffer Array of certain mesh from model
	/// </summary>
	/// <param name="index">index of mesh</param>
	/// <returns>Element Buffer Array</returns>
	virtual unsigned int getEboBuffer(unsigned int index) = 0;
};


/// <summary>
/// A model dedicated to objects in simulation which are managed in Instancing mechanism
/// </summary>
class InstancedModel : public Model
{
public:
	InstancedModel(InstancedObjectMesh& mesh, int instancesCount);

	void draw(const Shader* shader) const override;
	unsigned int getVboBuffer(unsigned int index) override;
	unsigned int getEboBuffer(unsigned int index) override;

	unsigned int getCudaOffsetBuffer();

protected:
	InstancedObjectMesh& mesh;
	unsigned int cudaOffsetBuffer;
	unsigned int instancesCount;
};

class SingleObjectModel : public Model
{
public:
	SingleObjectModel() {}
	SingleObjectModel(SingleObjectMesh& mesh) { addMesh(mesh); }

	void draw(const Shader* shader) const override;
	unsigned int getVboBuffer(unsigned int index) override;
	unsigned int getEboBuffer(unsigned int index) override;

	void addMesh(SingleObjectMesh& mesh);
protected:
	std::vector<SingleObjectMesh> meshes;
};

/// <summary>
/// A model dedicated to objects in simulation which exists in multiple units
/// </summary>
class MultipleObjectModel : public Model
{
public:
	MultipleObjectModel() { verticesInitialCount = objectCount = 0; }
	MultipleObjectModel(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, std::vector<glm::vec3>& initialPositions, unsigned int objectCount);

	void draw(const Shader* shader) const override;
	unsigned int getVboBuffer(unsigned int index) override;
	unsigned int getEboBuffer(unsigned int index) override;

	void addMesh(MultiObjectMesh& mesh);

protected:
	std::vector<MultiObjectMesh> meshes;
	unsigned int verticesInitialCount;
	unsigned int objectCount;
};

