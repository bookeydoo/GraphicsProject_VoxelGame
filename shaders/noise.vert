#version 330 core
layout(location=0) in vec3 aPos;
layout(location=2) in vec2 aTexCoord;

layout(location=3) in vec3 aInstancePos;
layout(location=4) in float aBlockType;
layout(location=5) in vec4 aBlockColor;

out vec2 TexCoord;
out float vBlockType;
out vec4 vBlockColor;

uniform sampler2D noiseTexture;
uniform float heightScale;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    TexCoord = aTexCoord;
    vBlockType = aBlockType;
    vBlockColor = aBlockColor;

    //Scale the texture
    vec2 scaledUV=(aInstancePos.xz)*0.1;

    vec3 worldPos = aPos + aInstancePos ;
    // worldPos.y += displacement*5.0;


    gl_Position = projection * view  * vec4(worldPos, 1.0);
}