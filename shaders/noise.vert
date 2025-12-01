#version 330 core

layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aTexCoord;
layout(location=2) in vec2 aVoxelCenterXZ;
layout(location=3) in float aBlockType;

out vec2 TexCoord;
out float vBlockType;


uniform sampler2D noiseTexture;
uniform float heightScale;
uniform mat4 view;
uniform mat4 projection;

void main() 
{
    TexCoord = aTexCoord;
    vBlockType = aBlockType;

    vec3 worldPos = aPos;

    // You can keep this code for additional effects if needed

    gl_Position = projection * view * vec4(worldPos, 1.0);
}