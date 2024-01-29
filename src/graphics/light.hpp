#pragma once

#include <glm/vec3.hpp>

using namespace glm;

/// <summary>
/// Represents light components
/// </summary>
struct Light
{
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

/// <summary>
/// Represents light components with direction
/// </summary>
struct DirLight : public Light
{
    vec3 direction;
};