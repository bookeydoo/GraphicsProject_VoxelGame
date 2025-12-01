#version 330 core

layout(location=0) in vec3 aPos;
layout(location=2) in vec2 aTexCoord;
layout(location=3) in float aBlockType;   // moved to location 3

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

    // World position (already baked into VBO)
    vec3 worldPos = aPos;

    // Sample noise using world XZ coordinates
    vec2 uv = worldPos.xz * 0.05;
    float noiseVal = texture(noiseTexture, uv).r;

    float displacement;
    if (noiseVal > 0.7) displacement = noiseVal * heightScale * 2.0;
    else if (noiseVal > 0.3) displacement = noiseVal * heightScale;
    else displacement = noiseVal * heightScale * 0.1;

    worldPos.y += displacement * 5.0;

    gl_Position = projection * view * vec4(worldPos, 1.0);
}