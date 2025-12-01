import os
import pyrr
from PIL import Image
import imgui
import pygame
import numpy as np
from imgui.integrations.opengl import ProgrammablePipelineRenderer
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from Classes.VBO import VBO 
from Classes.VAO import VAO
from Classes.EBO import EBO 
from Classes.Camera import Camera
from Classes.FrustumCull import FrustumCulling 
from Classes.WorldChunks import World 

WORLD_FORWARD = pyrr.Vector3([0.0, 0.0, -1.0])  # Negative Z-axis is typically "forward"
WORLD_BACKWARD = pyrr.Vector3([0.0, 0.0, 1.0])   # Positive Z-axis is "backward"
WORLD_RIGHT = pyrr.Vector3([1.0, 0.0, 0.0])      # Positive X-axis is "right"
WORLD_LEFT = pyrr.Vector3([-1.0, 0.0, 0.0])      # Negative X-axis is "left"
WORLD_UP = pyrr.Vector3([0.0, 1.0, 0.0])         # Positive Y-axis is "up"

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
 

def InitOpengl():
    # Request OpenGL 3.3 compatibility context (your imgui/Pygame renderer expects this)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_COMPATIBILITY
    )

def Compile_Shader(source, ShaderType):
    shader = glCreateShader(ShaderType)
    glShaderSource(shader, source)
    glCompileShader(shader)

    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise Exception(glGetShaderInfoLog(shader))
    return shader

def CreateShaderProgram(vertexShader, fragShader):
    program = glCreateProgram()
    vs = Compile_Shader(vertexShader, GL_VERTEX_SHADER)
    fs = Compile_Shader(fragShader, GL_FRAGMENT_SHADER)
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)

    if not glGetProgramiv(program, GL_LINK_STATUS):
        raise Exception(glGetProgramInfoLog(program))
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program

def load_texture(path):
    # Create texture ID
    textureID = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, textureID)

    # Texture wrapping options
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    # Texture filtering options
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Load image
    image = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
    img_data = image.convert("RGBA").tobytes()
    width, height = image.size

    # Upload image to OpenGL
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, img_data)

    glGenerateMipmap(GL_TEXTURE_2D)

    return textureID

def ExitFunc():
    pygame.event.set_grab(False)
    pygame.quit()

#IMGUI INPUT
def process_pygame_inputs():
    io = imgui.get_io()
    io.mouse_pos = pygame.mouse.get_pos()
    io.mouse_down[0] = pygame.mouse.get_pressed()[0]
    io.mouse_down[1] = pygame.mouse.get_pressed()[1]
    io.mouse_down[2] = pygame.mouse.get_pressed()[2]


def main():
    #pygame.mixer.pre_init(44100, -16, 2, 512)

    print(imgui.__version__)
    print(imgui.__file__)
    pygame.init()
    
    InitOpengl()

    display = (1600, 800)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Simple voxel test")

    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    glEnable(GL_DEPTH_TEST)

    # glEnable(GL_CULL_FACE)
    # glCullFace(GL_BACK)
    # glFrontFace(GL_CCW)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    #current directory
    BaseDir=os.path.dirname(os.path.abspath(__file__))

    #Import textures
    
    TexturesDir=os.path.join(BaseDir,"Resources")
    yeezytext=load_texture(TexturesDir+"/yeezus.jpg")

    sandtext=load_texture(TexturesDir+"/sandTexture.jpg")
    snowtext=load_texture(TexturesDir+"/snowTexture.jpg")
    dirttext=load_texture(TexturesDir+"/DirtTexture.jpg")

    #PERLIN NOISE TEXT
    PerlinNoise1=load_texture(TexturesDir+"/perlin_noise.png")

    
    #import shaders
    ShaderDir=os.path.join(BaseDir,"shaders")

    vertexPath=os.path.join(ShaderDir,"noise.vert")
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
    # VBO and VAO setup
    ######################################################

    VAO1 = VAO()
    VAO1.bind()

    # --- Cube Vertex Data ---
    CubeVBO = VBO(vertices=None,Static=False)
    CubeVBO_ID=CubeVBO.ID

    
    ebo = EBO(indices)
    ebo_ID=ebo.ID
    ebo.bind()

    stride = 8 * 4
    VAO1.LinkVBO(CubeVBO, 0, 3, GL_FLOAT, stride, 0)     # pos
    VAO1.LinkVBO(CubeVBO, 1, 3, GL_FLOAT, stride, 12)    # color 
    VAO1.LinkVBO(CubeVBO, 2, 2, GL_FLOAT, stride, 24)    # texcoord

    VAO1.unbind()
        


    # Setup ImGui
    imgui.create_context()
    renderer = ProgrammablePipelineRenderer()

    ######################################################
    #Matrix setup
    ######################################################

    camera=Camera(1600,800,Position=[0,9,0])

    glUseProgram(ShaderProgram)
    # Helpers to get uniform locations TODO : understand what these do
    model_loc = glGetUniformLocation(ShaderProgram, "model")
    view_loc = glGetUniformLocation(ShaderProgram, "view")
    proj_loc = glGetUniformLocation(ShaderProgram, "projection")
    
   
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

    #Init FRUSTUM
    MyFrustum=FrustumCulling()

    #INIT WORLD
    MyWorld=World()
    MyWorld.InitWorld(np.array(camera.Position,dtype=np.float32))
    
    


    while True:

        # Handle events
        for event in pygame.event.get():

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

                if event.key == pygame.K_r:
                    if not Rotated:
                        Rotated = True
                    else:
                        Rotated = False
        
        keys=pygame.key.get_pressed()
        if keys[pygame.K_w]:
            camera.Position += camera.Orientation* camera.speed
            print("W is pressed")
            print(camera.Position)

        if keys[pygame.K_s]:
            camera.Position -= camera.Orientation * camera.speed
            print("S is pressed")
            print(camera.Position)

        if keys[pygame.K_d]:
            camera.Position +=   camera.Right * camera.speed 
            print("d is pressed")
            print(camera.Position)

        if keys[pygame.K_a]:
            camera.Position -=  camera.speed * camera.Right 
            print("a is pressed")

        if keys[pygame.K_SPACE]:
            camera.Position += WORLD_UP * camera.speed 

        if keys[pygame.K_LCTRL]:
            camera.Position -= camera.Up * camera.speed 


            
        #DEBUGGING
        print("Pos:",camera.Position)


        #Compute mouse delta
        mouse_dx,mouse_dy=pygame.mouse.get_rel() 
        camera.inputs(mouse_dx,mouse_dy)

        camera.update_vectors()
    

        projection,view=camera.Matrix(FOVdeg=45.0,nearPlane=0.1,farPlane=500.0)

        glUniformMatrix4fv(view_loc,1,GL_FALSE,view)
        glUniformMatrix4fv(proj_loc,1,GL_FALSE,projection)

        MyFrustum.Extract_frustum_planes(projection,view)

        VAO1.bind()
 

        # -----------------------
        # Update ImGui screen size
        io=imgui.get_io()
        io.display_size=pygame.display.get_surface().get_size()
        io.display_fb_scale = (1.0, 1.0)
        
        process_pygame_inputs()

        imgui.new_frame()
        imgui.set_next_window_position(50,50)
        imgui.set_next_window_size(400,400)

        if ShowDevWindow==True:
            # UI (must have begin/end)
            imgui.begin("Debug Window")
            imgui.text("This is a test window")
            
            pos_x=camera.Position[0]
            pos_y=camera.Position[1]
            pos_z=camera.Position[2]

            imgui.text("--Position--")            
            imgui.text(f"x pos:{pos_x:.2f}")
            imgui.text(f"y pos:{pos_y:.2f}")
            imgui.text(f"z pos:{pos_z:.2f}")

            if imgui.button("Click me"):
             print("hello world")
            imgui.end()
            # -----------------------

        #Rotation logic
        rotation = pyrr.matrix44.create_from_axis_rotation(
            axis=[0.5, 1.0, 0.0], theta=np.radians(rotationAngle)
        ).astype(np.float32)

        translation = pyrr.matrix44.create_from_translation([0, 0, -8])
        model=pyrr.matrix44.multiply(translation,rotation)
            

        ######################################################
        #Rendering
        ######################################################

        #Clear 
        glClearColor(135/255, 206/255, 235/255, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        #Cube Textures
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D,dirttext)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D,sandtext)

        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D,snowtext)

        glActiveTexture(GL_TEXTURE4)
        glBindTexture(GL_TEXTURE_2D,yeezytext)


        
        #Perlin Noise Text
        glActiveTexture(GL_TEXTURE5)
        glBindTexture(GL_TEXTURE_2D,PerlinNoise1)

        glUniform1i(glGetUniformLocation(ShaderProgram,"noiseTexture"),5)
        glUniform1f(glGetUniformLocation(ShaderProgram,"heightScale"),2.0)

        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE,projection)
        glUniform1iv(glGetUniformLocation(ShaderProgram,"BlockTextures"),
                     4,np.array([0,1,2,3],dtype=np.int32))

        VAO1.bind()

        #Call world generation func
        MyWorld.DrawVisiChunks(MyFrustum,CubeVBO_ID,ebo_ID,VAO1.ID)


        VAO1.unbind()
        CubeVBO.unbind()


        # Render ImGui on top
        imgui.render()
        renderer.render(imgui.get_draw_data())

        pygame.display.flip()
        pygame.time.wait(10)


main()