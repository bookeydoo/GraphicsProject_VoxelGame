#version 330 core
in vec2 TexCoord;
in float vBlockType;
in vec4 vBlockColor;

out vec4 FragColor;

uniform sampler2D BlockTextures[13];

void main() {

    int id=int(vBlockType);

    if(id ==99){
        FragColor=vBlockColor;
    }
    else{
        int id = int(vBlockType);
        FragColor = texture(BlockTextures[id], TexCoord);
        
    }
}
