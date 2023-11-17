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

    Mesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, std::vector<Texture>&& textures);
    void draw(const Shader* shader, bool instanced = true) const;
    unsigned int getVBO();
    unsigned int getEBO();
    void setVertexOffsetAttribute();
private:
    //  render data
    unsigned int VAO, VBO, EBO;

    void setupMesh();
};