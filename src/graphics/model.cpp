#include "model.hpp"

#include "../meta_factory/blood_cell_factory.hpp"
#include "textures/texture_loading.hpp"

#include <glad/glad.h>
#include <iostream>
#include <vector>


#pragma region Model

void Model::draw(const Shader* shader) const
{
    for (Mesh mesh : meshes) {
        mesh.draw(shader);
    }
}

unsigned int Model::getVboBuffer(unsigned int index)
{
    return meshes[index].getVBO();
}

unsigned int Model::getEboBuffer(unsigned int index)
{
    return meshes[index].getEBO();
}
void Model::addMesh(Mesh& mesh)
{
    meshes.push_back(mesh);
}

#pragma endregion

#pragma region Instanced model

unsigned int InstancedModel::getCudaOffsetBuffer()
{
    return cudaOffsetBuffer;
}

InstancedModel::InstancedModel(Mesh& mesh, unsigned int instancesCount)
{
    Model::addMesh(mesh);
    this->instancesCount = instancesCount;
    glGenBuffers(1, &cudaOffsetBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, cudaOffsetBuffer);
    glBufferData(GL_ARRAY_BUFFER, instancesCount * sizeof(glm::vec3), NULL, GL_DYNAMIC_DRAW);

    // Set up vertex attrubute
    for (Mesh& mesh : meshes)
    {
        mesh.setVertexOffsetAttribute();
    }
}
#pragma endregion

#pragma region Multiple object model

MultipleObjectModel::MultipleObjectModel(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, std::vector<glm::vec3>& initialPositions, unsigned int objectCount)
{
    MultiObjectMesh mesh = MultiObjectMesh(std::move(vertices), std::move(indices), initialPositions, objectCount);
    Model::addMesh(mesh);
    this->objectCount = objectCount;
    this->verticesInitialCount = mesh.vertices.size();
}

#pragma endregion
