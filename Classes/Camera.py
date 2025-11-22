import pyrr
import numpy as np
from OpenGL.GL import *

class Camera:
    def __init__(self, width, height, Position=([0.0, 0.0, 3.0])):
        self.width = width
        self.height = height
        
        self.Position = np.array(Position,dtype=np.float32)
        self.Up = np.array([0.0, 1.0, 0.0],dtype=np.float32)
        
        # FPS camera angles
        self.yaw = -180.0     # looking forward (-Z)
        self.pitch = 0.0
        
        self.speed = 0.20
        self.sensitivity = 0.1

        
        self.update_vectors()   # compute Orientation, Right, Up

    # -----------------------------------------------------
    def update_vectors(self):
        # Convert yaw/pitch to directional vector
        yaw_r = np.radians(self.yaw)
        pitch_r = np.radians(self.pitch)

        front = np.array([
            np.cos(yaw_r) * np.cos(pitch_r),
            np.sin(pitch_r),
            np.sin(yaw_r) * np.cos(pitch_r)
        ],dtype=np.float32)

        self.Orientation = front/np.linalg.norm(front)

        cross_product=np.cross(self.Orientation,self.Up)
        self.Right = cross_product/np.linalg.norm(cross_product)

          


    # -----------------------------------------------------
    def Matrix(self, FOVdeg, nearPlane, farPlane):
        target = self.Position + self.Orientation
        view = pyrr.matrix44.create_look_at(self.Position, target, self.Up)
        proj = pyrr.matrix44.create_perspective_projection(
            FOVdeg, self.width / self.height, nearPlane, farPlane
        )
        return proj.astype(np.float32), view.astype(np.float32)

    # -----------------------------------------------------
    def inputs(self,  mouse_dx=0, mouse_dy=0):
        # Look
        if mouse_dx != 0 or mouse_dy != 0:
            self.yaw   += mouse_dx * self.sensitivity
            self.pitch -= mouse_dy * self.sensitivity

            # Clamp pitch to avoid camera flipping
            self.pitch = max(-89.0, min(89.0, self.pitch))

            self.update_vectors()