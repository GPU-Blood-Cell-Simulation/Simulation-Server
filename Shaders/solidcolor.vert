#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 3) in vec3 offset;

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

void main()
{
    mat4 modelWithPos =  mat4(vec4(1, 0, 0, 0), vec4(0, 1, 0, 0), vec4(0, 0, 1, 0), vec4(offset, 1.0)) * model; 
    gl_Position = projection * view * modelWithPos * vec4(aPos, 1.0); 
}