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
    // texCoords
    glm::vec2 texCoords;
};

struct Texture {
    unsigned int id;
    std::string type;
    std::string path;
};


class Mesh {
public:
    // mesh data
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture> textures;

    void draw(const Shader* shader) const;

    unsigned int getVBO();
    unsigned int getEBO();
    void setVertexOffsetAttribute();
protected:

    Mesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, std::vector<Texture>&& textures);
    //  render data
    unsigned int VAO, VBO, EBO;

    void setupMesh();
};

class SingleObjectMesh : public Mesh
{
public:
    SingleObjectMesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, std::vector<Texture>&& textures);
    SingleObjectMesh() : Mesh(std::move(std::vector<Vertex>()), std::move(std::vector<unsigned int>()), std::move(std::vector<Texture>())) {}

    void drawInstanced(const Shader* shader, unsigned int instancesCount) const;
};

class MultiObjectMesh : public Mesh
{
public:
    MultiObjectMesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, std::vector<Texture>&& textures, std::vector<glm::vec3>& initialPositions, unsigned int objectCount);
    void DuplicateObjects(std::vector<glm::vec3>& initialPositions);

protected:
    void prepareMultipleObjects(std::vector<glm::vec3>& initialPositions);
    unsigned int objectCount;
};