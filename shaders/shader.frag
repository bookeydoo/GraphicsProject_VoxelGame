#version 330 core
in vec2 TexCoord;
in float vBlockType;
out vec4 FragColor;

uniform sampler2D BlockTextures[4];

void main() {
    int id = int(vBlockType);
    vec4 tex = texture(BlockTextures[id], TexCoord);
    FragColor = tex;
}
