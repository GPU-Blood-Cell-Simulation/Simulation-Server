#version 410 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 3) in vec3 offset;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    mat4 modelWithPos =  mat4(vec4(1, 0, 0, 0), vec4(0, 1, 0, 0), vec4(0, 0, 1, 0), vec4(offset, 1.0)) * model; 
    gl_Position = projection * view * modelWithPos * vec4(aPos, 1.0);
    Normal = mat3(transpose(inverse(modelWithPos))) * aNormal;
    FragPos = vec3(modelWithPos * vec4(aPos, 1.0));
}