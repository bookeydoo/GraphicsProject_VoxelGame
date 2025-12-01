import numpy as np

class FrustumCulling:
    def __init__(self):
        self.planes=[]



    def Extract_frustum_planes(self, projection, view):

        self.planes = []

        # Row-major, row-vectors → view first, then projection
        m = view @ projection

        # Extract planes (using ROWS, not columns)
        # Left
        self.planes.append(m[3] + m[0])
        # Right
        self.planes.append(m[3] - m[0])
        # Bottom
        self.planes.append(m[3] + m[1])
        # Top
        self.planes.append(m[3] - m[1])
        # Near
        self.planes.append(m[3] + m[2])
        # Far
        self.planes.append(m[3] - m[2])

        # Normalize
        for i in range(6):
            n = np.linalg.norm(self.planes[i][:3])
            if n > 0:
                self.planes[i] = self.planes[i] / n

        return self.planes

    def aabb_visible(self, min_pos, max_pos):
        #Now self.planes contains 6 correctly normalized planes from the current frame

        if len(self.planes) != 6:
            print("Critical error")
            return True

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