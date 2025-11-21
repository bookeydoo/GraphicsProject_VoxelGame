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
        
        # Mouse look - only process if there's actual movement
        if mouse_dx != 0 or mouse_dy != 0:
            # Yaw (left-right rotation around Up axis)
            yaw_angle = np.radians(-mouse_dx * self.sensitivity)
            rot_y = pyrr.matrix44.create_from_axis_rotation(self.Up, yaw_angle)
            
            # Pitch (up-down rotation around Right axis)
            pitch_angle = np.radians(-mouse_dy * self.sensitivity)
            rot_x = pyrr.matrix44.create_from_axis_rotation(right, pitch_angle)
            
            # Apply rotations to orientation (yaw first, then pitch)
            orientation4 = np.array([*self.Orientation, 0.0], dtype=np.float32)  # 0.0 for direction vector
            orientation4 = pyrr.matrix44.apply_to_vector(rot_y, orientation4)
            orientation4 = pyrr.matrix44.apply_to_vector(rot_x, orientation4)
            
            # Extract the first 3 components
            self.Orientation = pyrr.Vector3(orientation4[:3])
            self.Orientation = pyrr.vector.normalize(self.Orientation)