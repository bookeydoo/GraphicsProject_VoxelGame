#version 330 core
//Position coords
layout (location = 0) in vec3 aPos;
//Color coords
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 aTexCoord;
layout (location = 3) in vec3 aOffset; 

//Outputs the color for the fragment shader
out vec3 ourColor;

//Outputs the texture coords to the fragment shader
//add it here

//Controls the scale of the vertices

out vec2 TexCoord;

uniform float scale;

uniform mat4 camMatrix;

void main()
{
    vec3 pos=aPos+aOffset;
    gl_Position =camMatrix *vec4(pos,1.0);
    ourColor = aColor;
    TexCoord=aTexCoord;
}