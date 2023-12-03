#include "shader.hpp"

#include <fstream>
#include <glad/glad.h> // include glad to get all the required OpenGL headers
#include <iostream>
#include <sstream>
#include <string_view>


std::string getShaderCode(const char* filePath)
{
    std::string shaderCode;
    std::ifstream shaderFile;
    // ensure ifstream objects can throw exceptions:
    shaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try
    {
        // open files
        shaderFile.open(filePath);
        std::stringstream vShaderStream, fShaderStream;
        // read file's buffer contents into streams
        vShaderStream << shaderFile.rdbuf();
        // close file handlers
        shaderFile.close();
        // convert stream into string
        shaderCode = vShaderStream.str();
    }
    catch (std::ifstream::failure e)
    {
        std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ, path: " << filePath << std::endl;
        exit(1);
    }
    return shaderCode;
}

GLuint compileShader(GLuint type, const char* shaderCode)
{
    int success;
    char infoLog[512];

    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &shaderCode, NULL);
    glCompileShader(shader);
    // print compile errors if any
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
        exit(1);
    };
    return shader;
}

Shader::Shader(const char* vertexPath, const char* fragmentPath)
{
    // 1. retrieve the vertex/fragment source code from filePath
    std::string vertexCode = getShaderCode(vertexPath);
    std::string fragmentCode = getShaderCode(fragmentPath);

    // 2. compile shaders
    GLuint vertex = compileShader(GL_VERTEX_SHADER, vertexCode.c_str());
    GLuint fragment = compileShader(GL_FRAGMENT_SHADER, fragmentCode.c_str());

    // 3. shader Program
    ID = glCreateProgram();
    glAttachShader(ID, vertex);
    glAttachShader(ID, fragment);
    glLinkProgram(ID);
    // print linking errors if any
    int success;
    char infoLog[512];

    glGetProgramiv(ID, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(ID, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    // 4. delete the shaders as they're linked into our program now and no longer necessary
    glDeleteShader(vertex);
    glDeleteShader(fragment);

}

void Shader::use()
{
    glUseProgram(ID);
}

void Shader::setFloat(const char* name, float value) const
{
    int location = glGetUniformLocation(ID, name);
    glUniform1f(location, value);
}

void Shader::setInt(const char* name, int value) const
{
    int location = glGetUniformLocation(ID, name);
    glUniform1i(location, value);
}

void Shader::setVector(const char* name, glm::vec3 vector) const
{
    int location = glGetUniformLocation(ID, name);
    glUniform3f(location, vector.x, vector.y, vector.z);
}

void Shader::setMatrix(const char* name, glm::mat4 matrix) const
{
    int location = glGetUniformLocation(ID, name);
    glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(matrix));
}

void Shader::setLighting(DirLight dirLight) const
{
    setVector("dirLight.direction", dirLight.direction);
    setVector("dirLight.ambient", dirLight.ambient);
    setVector("dirLight.diffuse", dirLight.diffuse);
    setVector("dirLight.specular", dirLight.specular);
}

SolidColorShader::SolidColorShader() : Shader("Shaders\\solidcolor.vert", "Shaders\\solidcolor.frag")
{}

GeometryPassShader::GeometryPassShader(unsigned int gBuffer) : Shader("Shaders\\geometry.vert", "Shaders\\geometry.frag"),
gBuffer(gBuffer)
{}

void GeometryPassShader::use()
{
    glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Shader::use();
}

PhongDeferredShader::PhongDeferredShader(unsigned int gPosition, unsigned int gNormal) : Shader("Shaders\\phong_deferred.vert", "Shaders\\phong_deferred.frag"),
gPosition(gPosition), gNormal(gNormal)
{}

void PhongDeferredShader::use()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Shader::use();

    // Activate textures that have been written to by the geometry pass shader
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gPosition);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, gNormal);
}

PhongForwardShader::PhongForwardShader() : Shader("Shaders\\phong_forward.vert", "Shaders\\phong_forward.frag")
{}

CylinderSolidColorShader::CylinderSolidColorShader() : Shader("Shaders\\cylindersolidcolor.vert", "Shaders\\cylindersolidcolor.frag")
{}

SpringShader::SpringShader() : Shader("Shaders\\spring.vert", "Shaders\\spring.frag")
{}
