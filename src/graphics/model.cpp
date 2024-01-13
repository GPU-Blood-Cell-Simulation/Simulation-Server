#include "model.hpp"

#include "../meta_factory/blood_cell_factory.hpp"
#include "textures/texture_loading.hpp"

#include <glad/glad.h>
#include <iostream>
#include <vector>

void SingleObjectModel::draw(const Shader* shader) const
{
    for(auto&& mesh : meshes)
        mesh.draw(shader);
}

unsigned int SingleObjectModel::getVboBuffer(unsigned int index)
{
    return meshes[index].getVBO();
}

unsigned int SingleObjectModel::getEboBuffer(unsigned int index)
{
    return meshes[index].getEBO();
}

void SingleObjectModel::addMesh(SingleObjectMesh& mesh)
{
    meshes.push_back(mesh);
}

void InstancedModel::draw(const Shader* shader) const
{
    mesh.draw(shader);
}

unsigned int InstancedModel::getCudaOffsetBuffer()
{
    return cudaOffsetBuffer;
}

unsigned int InstancedModel::getVboBuffer(unsigned int index)
{
    return mesh.getVBO();
}

unsigned int InstancedModel::getEboBuffer(unsigned int index)
{
    return mesh.getEBO();
}

InstancedModel::InstancedModel(InstancedObjectMesh& mesh, int instancesCount): mesh(mesh)
{
    this->instancesCount = instancesCount;
    glGenBuffers(1, &cudaOffsetBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, cudaOffsetBuffer);
    glBufferData(GL_ARRAY_BUFFER, instancesCount * sizeof(glm::vec3), NULL, GL_DYNAMIC_DRAW);

    mesh.setVertexOffsetAttribute();
}


MultipleObjectModel::MultipleObjectModel(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, std::vector<glm::vec3>& initialPositions, unsigned int objectCount)
{
    MultiObjectMesh mesh = MultiObjectMesh(std::move(vertices), std::move(indices), initialPositions, objectCount);
    addMesh(mesh);
    this->objectCount = objectCount;
    this->verticesInitialCount = mesh.vertices.size();
}


void MultipleObjectModel::draw(const Shader* shader) const
{
    for (auto&& mesh : meshes) {
        mesh.draw(shader);
    }
}

unsigned int MultipleObjectModel::getVboBuffer(unsigned int index)
{
    return meshes[index].getVBO();
}

unsigned int MultipleObjectModel::getEboBuffer(unsigned int index)
{
    return meshes[index].getEBO();
}

void MultipleObjectModel::addMesh(MultiObjectMesh& mesh)
{
    meshes.push_back(mesh);
}