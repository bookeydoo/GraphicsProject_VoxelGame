import pyrr
import numpy as np
from OpenGL.GL import *

class Camera:
    def __init__(self,width,height,Position=([0.0,0.0,3.0]),up=None):
        self.width= width
        self.height= height 
        self.Position=pyrr.Vector3([0.0,0.0,3.0]) if Position is None else pyrr.Vector3(Position)
        self.Up=pyrr.Vector3([0.0,1.0,0.0]) if up is None else pyrr.Vector3(up)
        self.Orientation=pyrr.vector3.create(0.0,0.0,-1.0,dtype=np.float32)
        self.speed=0.01
        self.sensitivity=0.2
        self.lastX=width/2
        self.lastY=height/2


    
    def Matrix(self,FOVdeg,nearPlane,farPlane):

       target=self.Position+self.Orientation
       view=pyrr.matrix44.create_look_at(self.Position,target,self.Up)

       proj=pyrr.matrix44.create_perspective_projection(FOVdeg,self.width/self.height,nearPlane,farPlane)

       return proj.astype(np.float32),view.astype(np.float32)
    
    def inputs(self, keys, mouse_dx=0.0, mouse_dy=0.0):
        """
        keys: a dictionary or set of pressed keys
        mouse_dx, mouse_dy: movement deltas since last frame
        """
        # Movement
        forward = pyrr.vector.normalize(self.Orientation)
        right = pyrr.vector.normalize(pyrr.vector3.cross(forward, self.Up))

        if keys.get("W"):  # Forward
            self.Position += forward * self.speed
        if keys.get("S"):  # Backward
            self.Position -= forward * self.speed
        if keys.get("A"):  # Left
            self.Position -= right * self.speed
        if keys.get("D"):  # Right
            self.Position += right * self.speed
        if keys.get("SPACE"):  # Up
            self.Position += self.Up * self.speed
        if keys.get("CTRL"):  # Down
            self.Position -= self.Up * self.speed

        # Mouse look
        rot_x = pyrr.matrix44.create_from_axis_rotation(right, np.radians(-mouse_dy * self.sensitivity))
        rot_y = pyrr.matrix44.create_from_axis_rotation(self.Up, np.radians(-mouse_dx * self.sensitivity))

        # Apply rotation to orientation
        orientation4 = np.append(self.Orientation, 1.0)  # Convert to 4D vector for matrix mul
        orientation4 = rot_x @ orientation4
        orientation4 = rot_y @ orientation4
        self.Orientation = pyrr.vector3.create(*orientation4[:3], dtype=np.float32)
        self.Orientation = pyrr.vector.normalize(self.Orientation)

       