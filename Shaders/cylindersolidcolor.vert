#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 3) in vec3 offset;

uniform mat4 projection;
uniform mat4 view;


void main()
{
    gl_Position = projection * view * vec4(aPos, 1.0); 
}