import numpy as np

class FrustumCulling:
    def __init__(self):
        self.planes=[]




    def Extract_frustum_planes(self,projection,view):

        self.viewPlaneMatrix = projection @ view

        # Left
        self.planes.append(self.viewPlaneMatrix[3] + self.viewPlaneMatrix[0])
        # Right
        self.planes.append(self.viewPlaneMatrix[3] - self.viewPlaneMatrix[0])
        # Bottom
        self.planes.append(self.viewPlaneMatrix[3] + self.viewPlaneMatrix[1])
        # Top
        self.planes.append(self.viewPlaneMatrix[3] - self.viewPlaneMatrix[1])
        # Near
        self.planes.append(self.viewPlaneMatrix[3] + self.viewPlaneMatrix[2])
        # Far
        self.planes.append(self.viewPlaneMatrix[3] - self.viewPlaneMatrix[2])

        Normalized=[]

        for p in self.planes:
            n=p[:3]
            l=np.linalg.norm(n)
            Normalized.append(p/l)
        
        self.planes=Normalized
        
        

        return self.planes 


    def aabb_visible(self, min_pos, max_pos):
        # Now self.planes contains 6 correctly normalized planes from the current frame
        for plane in self.planes:
            A, B, C, D = plane  # (A, B, C) is the plane normal, D is the distance

            # Optimization: Find the positive vertex (P-Vertex)
            # This is the vertex of the AABB closest to the plane's normal.
            px = max_pos[0] if A >= 0 else min_pos[0]
            py = max_pos[1] if B >= 0 else min_pos[1]
            pz = max_pos[2] if C >= 0 else min_pos[2]

            # Test P-Vertex distance to the plane
            # If distance < 0, the entire AABB is behind the plane and culled.
            if A * px + B * py + C * pz + D < 0:
                return False

        # If it passed all 6 plane checks, it is visible.
        return True