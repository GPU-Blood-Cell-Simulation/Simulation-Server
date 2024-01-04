#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "mesh.hpp"
#include "shader.hpp"


/// <summary>
/// General model class containing basic model properties
/// </summary>
class Model
{
public:
	Model() {};

	/// <summary>
	/// Calls OpenGL render pipeline for underlying meshes
	/// </summary>
	/// <param name="shader">pointer to shader passed to the pipeline</param>
	void draw(const Shader* shader) const;

	/// <summary>
	/// Gets Vertex Buffer Object of certain mesh from model
	/// </summary>
	/// <param name="index">index of mesh</param>
	/// <returns>Vertex Buffer Object</returns>
	unsigned int getVboBuffer(unsigned int index);
	
	/// <summary>
	/// Gets Element Buffer Array of certain mesh from model
	/// </summary>
	/// <param name="index">index of mesh</param>
	/// <returns>Element Buffer Array</returns>
	unsigned int getEboBuffer(unsigned int index);

	/// <summary>
	/// Adds external mesh to model
	/// </summary>
	/// <param name="mesh">new mesh</param>
	void addMesh(Mesh& mesh);

protected:

	std::string directory;
	std::vector<Mesh> meshes;
};


/// <summary>
/// A model dedicated to objects in simulation which are managed in Instancing mechanism
/// </summary>
class InstancedModel : public Model
{
public:
	InstancedModel(Mesh& mesh, unsigned int instancesCount);
	unsigned int getCudaOffsetBuffer();

protected:

	unsigned int cudaOffsetBuffer;
	unsigned int instancesCount;
};

/// <summary>
/// A model dedicated to objects in simulation which exists in multiple units
/// </summary>
class MultipleObjectModel : public Model
{
public:
	MultipleObjectModel() : Model() { verticesInitialCount = objectCount = 0; }
	MultipleObjectModel(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, std::vector<glm::vec3>& initialPositions, unsigned int objectCount);

protected:
	unsigned int verticesInitialCount;
	unsigned int objectCount;
};

