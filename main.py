import pygame
import os
import math
import imgui
from imgui.integrations.pygame import PygameRenderer
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

####################################################################################
        ###Defining the voxel shape###
####################################################################################
# Define vertices (corners)
vertices = [
    [2, 1, -1],[2, -1, -1],[-1, -1, -1],[-1, 1, -1],
    [2, 1, 1] ,[2, -1, 1] ,[-1, -1, 1] ,[-1, 1, 1]
]

# Define edges (lines between vertices)
edges = [
    (0,1), (1,2), (2,3), (3,0),
    (4,5), (5,6), (6,7), (7,4),
    (0,4), (1,5), (2,6), (3,7)
]

# Define faces (polygons)
surfaces = [
    (0,1,2,3), (4,5,6,7), (0,1,5,4),
    (2,3,7,6), (1,2,6,5), (0,3,7,4)
]

def ExitFunc():
    pygame.quit()

 
def main():
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()

    display = (1600, 800)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Simple voxel test")

    # Setup ImGui
    imgui.create_context()
    renderer = PygameRenderer()

    # OpenGL setup
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glShadeModel(GL_SMOOTH)

    CameraPos = [0, -3, -20]

    gluPerspective(45, display[0] / display[1], 0.1, 50.0)
    glTranslatef(*CameraPos)
    glRotatef(45, 1, 0, 0)

    Rotated = True

    while True:

        # Handle events
        for event in pygame.event.get():
            renderer.process_event(event)
            if event.type == pygame.QUIT:
                ExitFunc
                break

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    ExitFunc()
                    break

                if event.key == pygame.K_d:
                    CameraPos[0] += 0.5
                if event.key == pygame.K_a:
                    CameraPos[0] -= 0.5

                if event.key == pygame.K_r:
                    if not Rotated:
                        glRotatef(45, 1, 0, 0)
                        Rotated = True
                    else:
                        glRotatef(-45, 1, 0, 0)
                        Rotated = False

        # -----------------------
        # Update ImGui screen size
        w, h = pygame.display.get_surface().get_size()
        imgui.get_io().display_size = (w, h)

        imgui.new_frame()

        # UI (must have begin/end)
        imgui.begin("Debug Window")
        imgui.text("This is a test window")
        if imgui.button("Click me"):
            print("hello world")
        imgui.end()
        # -----------------------

        # Clear screen *before* ImGui
        glClearColor(135/255, 206/255, 235/255, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # (Draw your 3D scene here...)

        # Render ImGui on top
        imgui.render()
        renderer.render(imgui.get_draw_data())

        pygame.display.flip()
        pygame.time.wait(10)

    # Shutdown outside loop


main()