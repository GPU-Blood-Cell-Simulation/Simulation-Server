#pragma once

#include <glm/vec3.hpp>

using namespace glm;

struct Light
{
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct DirLight : public Light
{
    vec3 direction;
};