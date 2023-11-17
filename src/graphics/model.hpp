#pragma once

#include <assimp/scene.h>
#include <memory>
#include <string>
#include <vector>

#include "mesh.hpp"
#include "shader.hpp"


class Model
{
public:
    Model(const char* path);
    Model(Mesh mesh);

    void draw(const Shader* shader, bool instanced = true) const;
    unsigned int getCudaOffsetBuffer();
    Mesh getTopMesh();
    unsigned int getTopVboBuffer();
    unsigned int getTopEboBuffer();
protected:
    // Array of translation vectors for each instance - cuda writes to this
    unsigned int cudaOffsetBuffer;

    std::vector<Texture> textures_loaded;
    std::string directory;
    std::vector<Mesh> meshes;

    void loadModel(std::string path);
    void processNode(aiNode* node, const aiScene* scene);
    Mesh processMesh(aiMesh* mesh, const aiScene* scene);

    std::vector<Texture> loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName);

};