#version 330 core

layout (location = 0) in vec3 aPos;        // vertex position
layout (location = 1) in vec3 aColor;      // vertex color
layout (location = 2) in vec2 aTexCoord;   // vertex tex coord
layout (location = 3) in vec3 instanceOffset; // new: per-instance position

out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    // Add the instance offset to the vertex position
    vec4 worldPos = vec4(aPos + instanceOffset, 1.0);

    gl_Position = projection * view * model * worldPos;

    TexCoord = aTexCoord;
}
