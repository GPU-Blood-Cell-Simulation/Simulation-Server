#pragma once

#include "shader.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <memory>
#include <vector>


struct Vertex {
    // position
    glm::vec3 position;
    // normal
    glm::vec3 normal;
};

class Mesh {
public:
    // mesh data
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    virtual void draw(const Shader* shader) const;

    unsigned int getVBO();
    unsigned int getEBO();
    void setVertexOffsetAttribute();
protected:
    Mesh() {}
    Mesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices);
    //  render data
    unsigned int VAO, VBO, EBO;

    void setupMesh();
};

class SingleObjectMesh : public Mesh
{
public:
    SingleObjectMesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices);
    SingleObjectMesh() : Mesh(std::move(std::vector<Vertex>()), std::move(std::vector<unsigned int>())) {}
    void draw(const Shader* shader) const override;
};

class InstancedObjectMesh : public SingleObjectMesh
{
    int instancesCount;
public:
    InstancedObjectMesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, int instancedCount) :
        SingleObjectMesh(move(vertices), move(indices)) {
        this->instancesCount = instancedCount;
    };
    InstancedObjectMesh(int instancesCount) : SingleObjectMesh() { this->instancesCount = instancesCount; }

    void draw(const Shader* shader) const override;

};

class MultiObjectMesh : public Mesh
{
public:
    MultiObjectMesh() { objectCount = 0; };
    MultiObjectMesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, std::vector<glm::vec3>& initialPositions, unsigned int objectCount);
    void draw(const Shader* shader) const override;
    void DuplicateObjects(std::vector<glm::vec3>& initialPositions);

protected:
    void prepareMultipleObjects(std::vector<glm::vec3>& initialPositions);
    unsigned int objectCount;
};