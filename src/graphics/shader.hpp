#pragma once

#include "light.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <string>
#include <vector>

/// <summary>
/// Represents shader for rendering purposes
/// https://learnopengl.com/Getting-started/Shaders
/// </summary> 
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

class SpheresSolidColorShader : public Shader
{
public:
    SpheresSolidColorShader();
};

class SpheresPhongForwardShader : public Shader
{
public:
    SpheresPhongForwardShader();
};

class GeometryPassShader : public Shader
{
public:
    GeometryPassShader();
};

class PhongDeferredShader : public Shader
{
public:
    PhongDeferredShader();
};

class VeinSolidColorShader : public Shader
{
public:
    VeinSolidColorShader();
};

class SpringShader : public Shader
{
public:
    SpringShader();
};