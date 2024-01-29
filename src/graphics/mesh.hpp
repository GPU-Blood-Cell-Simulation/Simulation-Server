#pragma once

#include "shader.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <memory>
#include <vector>

/// <summary>
/// Represents visualization vertex
/// </summary>
struct Vertex {
    // position
    glm::vec3 position;
    // normal
    glm::vec3 normal;
};

/// <summary>
/// General mesh class containing basic mesh properties
/// </summary>
class Mesh {
public:
    // mesh data
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    /// <summary>
    /// Calls OpenGL render pipeline for mesh data
    /// </summary>
    /// <param name="shader">shader passed to OpenGL pipeline</param>
    virtual void draw(const Shader* shader) const;

    /// <summary>
    /// Gets mesh Vertex Buffer Object
    /// </summary>
    /// <returns>Vertex Buffer Object</returns>
    unsigned int getVBO();

    /// <summary>
    /// Gets mesh Element Array Buffer
    /// </summary>
    /// <returns>Element Array Buffer</returns>
    unsigned int getEBO();
    void setVertexOffsetAttribute();
protected:
    Mesh() {}
    Mesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices);
    //  render data
    unsigned int VAO, VBO, EBO;

    void setupMesh();
};

/// <summary>
/// A mesh dedicated to individual objects in simulation
/// </summary>
class SingleObjectMesh : public Mesh
{
public:
    SingleObjectMesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices);
    SingleObjectMesh() : Mesh(std::move(std::vector<Vertex>()), std::move(std::vector<unsigned int>())) {}
};

/// <summary>
/// A mesh dedicated to objects in simulation which are managed in Instancing mechanism
/// </summary>
class InstancedObjectMesh : public SingleObjectMesh
{
    int instancesCount;
public:
    InstancedObjectMesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, int instancedCount) :
        SingleObjectMesh(move(vertices), move(indices)) {
        this->instancesCount = instancedCount;
    };
    InstancedObjectMesh(int instancesCount) : SingleObjectMesh() { this->instancesCount = instancesCount; }

    /// <summary>
    /// Calls OpenGL render pipeline for mesh data
    /// </summary>
    /// <param name="shader">shader passed to OpenGL pipeline</param>
    void draw(const Shader* shader) const override;

};

/// <summary>
/// A mesh dedicated to objects in simulation which exists in multiple units
/// </summary>
class MultiObjectMesh : public Mesh
{
public:
    MultiObjectMesh() { objectCount = 0; };
    MultiObjectMesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, std::vector<glm::vec3>& initialPositions, unsigned int objectCount);
    
    /// <summary>
    /// Extends initial mesh with object duplicates based on initial positions
    /// </summary>
    /// <param name="initialPositions">collection of initial positions for duplicated objects</param>
    void DuplicateObjects(std::vector<glm::vec3>& initialPositions);

protected:
    void prepareMultipleObjects(std::vector<glm::vec3>& initialPositions);
    unsigned int objectCount;
};