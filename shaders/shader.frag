#version 330 core
in vec2 TexCoord;
in float vBlockType;
out vec4 FragColor;

uniform sampler2D BlockTextures[4];

void main() {
    int id = int(vBlockType);
    FragColor = texture(BlockTextures[id],TexCoord) ;
    // FragColor = vec4(TexCoord,0.0,1.0) ;
}
