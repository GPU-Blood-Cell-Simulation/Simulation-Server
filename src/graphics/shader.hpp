#pragma once

#include "light.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <string>
#include <vector>


// https://learnopengl.com/Getting-started/Shaders
class Shader
{
public:
    // the program ID
    unsigned int ID;
    // use/activate the shader
    virtual ~Shader() = default;
    virtual void use();
    // utility uniform functions
    void setFloat(const char* name, float value) const;
    void setInt(const char* name, int value) const;
    void setVector(const char* name, glm::vec3 vector) const;
    void setMatrix(const char* name, glm::mat4 matrix) const;
    void setLighting(DirLight dirLight) const;
protected:
    // constructor reads and builds the shader
    Shader(const char* vertexPath, const char* fragmentPath);
};

class SolidColorShader : public Shader
{
public:
    SolidColorShader();
};

class PhongForwardShader : public Shader
{
public:
    PhongForwardShader();
};

class GeometryPassShader : public Shader
{
public:
    GeometryPassShader(unsigned int gBuffer);
    void use() override;
private:
    unsigned int gBuffer;
};

class PhongDeferredShader : public Shader
{
public:
    PhongDeferredShader(unsigned int gPosition, unsigned int gNormal);
    void use() override;
private:
    unsigned int gPosition;
    unsigned int gNormal;
};

class CylinderSolidColorShader : public Shader
{
public:
    CylinderSolidColorShader();
};

class SpringShader : public Shader
{
public:
    SpringShader();
};