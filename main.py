import os
import pyrr
import math
import imgui
import pygame
import numpy as np
from imgui.integrations.pygame import PygameRenderer
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from VBO import VBO 
from VAO import VAO
from EBO import EBO 



def InitOpengl():
     # Request OpenGL 3.3 Core context
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_COMPATIBILITY
    )


def Compile_Shader(source,ShaderType):
    shader=glCreateShader(ShaderType)
    glShaderSource(shader,source)
    glCompileShader(shader)

    if glGetShaderiv(shader,GL_COMPILE_STATUS) != GL_TRUE:
        raise Exception(glGetShaderInfoLog(shader))
    return shader

def CreateShaderProgram(vertexShader,fragShader):
    program=glCreateProgram()
    vs=Compile_Shader(vertexShader,GL_VERTEX_SHADER)
    fs=Compile_Shader(fragShader,GL_FRAGMENT_SHADER)
    glAttachShader(program,vs)
    glAttachShader(program,fs)
    glLinkProgram(program)

    if not glGetProgramiv(program,GL_LINK_STATUS):
        raise Exception(glGetProgramInfoLog(program))
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program


def ExitFunc():
    pygame.quit()


####################################################################################
        ###Defining the voxel shape###
####################################################################################
# Define vertices (corners)
vertices = np.array([
    
	# positions		 #colors          #TexCoords U,V		
	# front face
	-0.5, -0.5, 0.5,  0.5, 0.2, 0.0,0.0,0.0,			# bottom left front(red) 0<
	-0.5,  0.5, 0.5,  0.0, 1.0, 0.0,0.0,1.0,			#top left   front (green) 1 <
	 0.5, -0.5, 0.5,  0.2, 0.3, 1.0,1.0,0.0,			 # bottom right front (blue) 2 <
	 0.5,  0.5, 0.5,  1.0, 1.0, 0.0,1.0,1.0,			# top right  front (yellow) 3<

	# back face
	-0.5,-0.5,-0.5,  1.0, 1.0, 0.0,	0.0,0.0,			#/ top right back(yellow)		4 
	-0.5, 0.5,-0.5,  1.0, 0.2, 1.0,	0.0,1.0,			 #/ top right  back(yellow)		5 
	 0.5, 0.5,-0.5,  1.0, 0.4, 0.0,	1.0,1.0,			 #// top right  back(yellow)	6
	 0.5,-0.5,-0.5,  1.0, 0.3, 1.0,	0.0,1.0,			#/ top right  back(yellow)		7	
	# Top face
	-0.5,  0.5, 0.5,  1.0, 1.0, 0.0,0.0,0.0,			#/ top right back(yellow)1<		8
	 0.5,  0.5, 0.5,  1.0, 0.2, 1.0,1.0,0.0,			#// top right  back(yellow)3	10
	-0.5,  0.5,-0.5,  1.0, 0.4, 0.0,0.0,1.0,			# // top right  back(yellow)5	9
	 0.5,  0.5,-0.5,  1.0, 0.3, 1.0,1.0,1.0,			#/ top right  back(yellow)6		11
	
	# bottom face
	-0.5, -0.5, 0.5,  1.0, 1.0, 0.0, 	0.0,1.0,			#/ top right back(yellow)0<		12
 	 0.5, -0.5,-0.5,  1.0, 0.2, 1.0,	1.0,1.0,			 #/ top right  back(yellow)2	13
	-0.5, -0.5, 0.5,  1.0, 0.4, 0.0,	0.0,0.0,			  #/ top right  back(yellow)4	14
	 0.5, -0.5,-0.5,  1.0, 0.3, 1.0, 	0.0,1.0,			#/ top right  back(yellow)7		15
	
	#right face
	 0.5, -0.5, 0.5,  1.0, 1.0, 0.0, 	0.0,0.0,			#/ top right back(yellow)2<		16
	 0.5,  0.5, 0.5,  1.0, 0.4, 0.0,	0.0,1.0,			  #/ top right  back(yellow)3	17
	 0.5,  0.5,-0.5,  1.0, 0.2, 1.0,	1.0,1.0,			 #/ top right  back(yellow)6	18
	 0.5, -0.5,-0.5,  1.0, 0.3, 1.0, 	1.0,0.0,			#/ top right  back(yellow)7		19
	
	#/left face
	-0.5, -0.5, 0.5, 1.0, 0.2, 1.0,	1.0,0.0,			 #/ top right  back(yellow)0	20
	-0.5,  0.5, 0.5, 1.0, 0.4, 0.0,	1.0,1.0,			  #/ top right  back(yellow)1	21
	-0.5,  0.5,-0.5, 1.0, 1.0, 0.0, 0.0,1.0,			#  top right back(yellow)5<		22
	-0.5, -0.5,-0.5, 1.0, 0.3, 1.0, 0.0,0.0			#top right  back(yellow)4		23
	

],dtype=np.float32)

indices =np.array ([
	#/ Front face (0,1,2,3)
	0, 1, 2,
	2, 3, 1,

	#/ Back face (4,5,6,7)
	4, 5, 6,
	6, 7, 4,

	#/ Top face (8,9,10,11)
	8, 10, 11,
	8, 9, 11,

	#/ Bottom face (12,13,14,15)
	12, 14, 15,
	12, 13, 15,

	#/ Right face (16,17,18,19)
	16, 17, 18,
	18, 19, 16,

	#/ Left face (20,21,22,23)
	20, 21, 22,
	22, 23, 20
],dtype=np.uint32)

# Define edges (lines between vertices)
#edges = np.array([
#    (0,1), (1,2), (2,3), (3,0),
#    (4,5), (5,6), (6,7), (7,4),
#    (0,4), (1,5), (2,6), (3,7)
#],dtype=np.float32)

# Define faces (polygons)
#surfaces = np.array([
#    (0,1,2,3), (4,5,6,7), (0,1,5,4),
#    (2,3,7,6), (1,2,6,5), (0,3,7,4)
#],dtype=np.float32)
 

def main():
    #pygame.mixer.pre_init(44100, -16, 2, 512)

    print(imgui.__version__)
    print(imgui.__file__)
    pygame.init()
    
    InitOpengl()

    display = (1600, 800)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Simple voxel test")

    glEnable(GL_DEPTH_TEST)

    #import shaders
    BaseDir=os.path.dirname(os.path.abspath(__file__))
    ShaderDir=os.path.join(BaseDir,"shaders")

    vertexPath=os.path.join(ShaderDir,"shader.vert")
    fragPath=os.path.join(ShaderDir,"shader.frag")

    with open(vertexPath, 'r') as f:
        vertexSrc= f.read()

    with open(fragPath, 'r') as f:
        fragmentSrc= f.read()
    
    ######################################################
    #Create Shader
    #######################################################

    ShaderProgram=CreateShaderProgram(vertexSrc,fragmentSrc)
    
    ######################################################
    #VBO and VAOS
    ######################################################

    #1st VAOs
    VAO1=VAO()
    VAO1.bind()

    #1.VBO for cube geometry
    CubeVBO=VBO(vertices)

    #2.VBO for instance positions 
    InstanceVBO=VBO()
    InstanceVBO_ID=InstanceVBO.ID

    #EBO
    ebo=EBO(indices)

    #stride =8*4 floats
    stride=8*4

    VAO1.LinkVBO(CubeVBO,0,3,GL_FLOAT,stride,0)
    VAO1.LinkVBO(CubeVBO,1,3,GL_FLOAT,stride,12)
    VAO1.LinkVBO(CubeVBO,2,2,GL_FLOAT,stride,24)

    VAO1.unbind()
    

    # Setup ImGui
    imgui.create_context()
    renderer = PygameRenderer()

    ######################################################
    #Matrix setup
    ######################################################

    view= pyrr.matrix44.create_look_at(
        eye=[0,0,3],target=[0,0,0],up=[0,1,0]
    ).astype(np.float32)
    projection= pyrr.matrix44.create_perspective_projection(
        fovy=45.0,aspect=display[0]/display[1],near=0.1,far=50.0
    ).astype(np.float32)

    # Helpers to get uniform locations TODO : understand what these do
    model_loc = glGetUniformLocation(ShaderProgram, "model")
    view_loc = glGetUniformLocation(ShaderProgram, "view")
    proj_loc = glGetUniformLocation(ShaderProgram, "projection")
    
    
    #init Camera
    CameraPos = [0, -3, -20]
    


    #Element 

    rotationAngle=0.0
    #Positons 

    LightSourcePos=[0,20,0] 
    PlayerPos=[0,0,0] 

    #Bool for checks 
    Rotated=True 

    #Timers 

    #Debug vars 
    ShowDevWindow=True



    while True:

        # Handle events
        for event in pygame.event.get():
            renderer.process_event(event)
            if event.type == pygame.QUIT:
                ExitFunc()

            if event.type == VIDEORESIZE:
                glViewport(0,0,event.w,event.h)
                projection = pyrr.matrix44.create_perspective_projection(
                    45.0, event.w/event.h, 0.1, 100.0).astype(np.float32)

            if event.type == pygame.KEYDOWN:
                if event.key ==pygame.K_F12:
                    if not ShowDevWindow:
                        print("Enabled Dev window")
                        ShowDevWindow=True
                    else:
                        print("Disabled Dev window")
                        ShowDevWindow=False
                if event.key == pygame.K_ESCAPE:
                    ExitFunc()
                    break

                if event.key == pygame.K_d:
                    CameraPos[0] += 0.5
                if event.key == pygame.K_a:
                    CameraPos[0] -= 0.5

                if event.key == pygame.K_r:
                    if not Rotated:
                        Rotated = True
                    else:
                        Rotated = False

        # -----------------------
        # Update ImGui screen size
        w, h = pygame.display.get_surface().get_size()
        imgui.get_io().display_size = (w, h)

        renderer.process_event(event)
        imgui.new_frame()

        if ShowDevWindow==True:
            # UI (must have begin/end)
            imgui.begin("Debug Window")
            imgui.text("This is a test window")
            if imgui.button("Click me"):
             print("hello world")
            imgui.end()
            # -----------------------

        #Rotation logic
        rotationAngle+=1.0 
        model=pyrr.matrix44.create_from_axis_rotation(
            axis=[0.5,1.0,0.0],theta=np.radians(rotationAngle)
        ).astype(np.float32)

        ######################################################
        #Rendering
        ######################################################

        #Clear 
        glClearColor(135/255, 206/255, 235/255, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(ShaderProgram)

        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

        VAO1.bind()
        #Draw elements
        glDrawElements(GL_TRIANGLES,len(indices),GL_UNSIGNED_INT,None)
        VAO1.unbind()


        # Render ImGui on top
        imgui.render()
        renderer.render(imgui.get_draw_data())

        pygame.display.flip()
        pygame.time.wait(10)


main()