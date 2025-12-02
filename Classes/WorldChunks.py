from dataclasses import dataclass,field
from OpenGL.GL import *
import numpy as np
import math

Chunk_Size=4
Voxel_Count=Chunk_Size**3

@dataclass
class Voxel:
    BlockType: np.uint32=np.uint32(0)


@dataclass
class Chunk:
    Position:  np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    Size: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    Voxels: np.ndarray = field(default_factory=lambda: np.zeros(Voxel_Count, dtype=np.uint32))

class World:

	#std::unordered_map<glm::vec3, Chunk*> worldMap;
	#std::vector<glm::vec3> visibleCubePositions;
    def __init__(self):
        self.WorldMap: dict[tuple,Chunk]={}
        self.VisibleCubePositions: list[tuple]=[]
        self.BlockTypes=[]
      
    def InitWorld(self,PlayerPos):
        radius = 4

        for cx in range(-radius,radius+1):
            for cz in range(-radius,radius+1):

                #allocate new chunk like in c++
                chunk = Chunk()

                # integer chunk coordinates
                chunk.Position = np.array([cx, 0, cz], dtype=np.float32)

                # all chunks are the same size
                chunk.Size = np.array([Chunk_Size]*3, dtype=np.float32)

                for i in range(Voxel_Count):
                    if cz<0:
                        chunk.Voxels[i]=1
                    elif cz>0:
                        chunk.Voxels[i]=2
                    else:
                        chunk.Voxels[i]=3



                #Insert into Hashmap(dict)
                key=(cx,0,cz)
                self.WorldMap[key]=chunk
            
    def DrawVisiChunks(self, instanceVBO_ID, EBO_ID):
        self.VisibleCubePositions.clear()
        self.BlockTypes.clear()

        #  Collect positions and block types
        for key, chunk in self.WorldMap.items():
            chunk_offset = chunk.Position * Chunk_Size

            for z in range(Chunk_Size):
                for y in range(Chunk_Size):
                    for x in range(Chunk_Size):
                        index = x + y * Chunk_Size + z * Chunk_Size * Chunk_Size
                        voxel = chunk.Voxels[index]

                        local_pos = np.array([x, y, z], dtype=np.float32)
                        world_pos = chunk_offset + local_pos

                        self.VisibleCubePositions.append(world_pos)
                        self.BlockTypes.append(chunk.Voxels[index])

        # Upload instance data to GPU (after loops)
        if self.VisibleCubePositions:
            instance_data = np.zeros((len(self.VisibleCubePositions), 4), dtype=np.float32)
            instance_data[:, :3] = self.VisibleCubePositions
            instance_data[:, 3] = np.array(self.BlockTypes,dtype=np.float32) 

            glBindBuffer(GL_ARRAY_BUFFER, instanceVBO_ID)
            glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)

            #  Draw instanced
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_ID)
            glDrawElementsInstanced(
                GL_TRIANGLES,
                36,  # number of indices per cube
                GL_UNSIGNED_INT,
                None,
                len(self.VisibleCubePositions)
            )
    
    def DrawVoxel(self,worldPos:np.ndarray):
       # Compute integer world coordinates (e.g., floor(2.5) -> 2, floor(-1.5) -> -2)
        wx, wy, wz = map(int, np.floor(worldPos))

        # 1. Compute chunk coordinates (correctly handles negative floor division)
        cx, cy, cz = wx // Chunk_Size, wy // Chunk_Size, wz // Chunk_Size
        chunk_coords = (cx, cy, cz)

        if chunk_coords not in self.WorldMap:
            # If the chunk isn't loaded, we can't place a block
            # For debugging, you might print a message here
            return

        chunk = self.WorldMap[chunk_coords]

        # 2. Compute local voxel coordinates (0 to Chunk_Size - 1)
        # This is the coordinate relative to the chunk's origin (cx*Chunk_Size, cy*Chunk_Size, cz*Chunk_Size)
        # This formulation avoids potential negative index issues from the raw modulo operator in some contexts.
        lx = wx - (cx * Chunk_Size)
        ly = wy - (cy * Chunk_Size)
        lz = wz - (cz * Chunk_Size)

        # 3. Compute the 1D voxel index
        voxel_index = lx + ly * Chunk_Size + lz * Chunk_Size * Chunk_Size

        # Safety check:
        if not (0 <= voxel_index < Voxel_Count):
             print(f"Error: Voxel Index {voxel_index} out of bounds for world pos ({wx}, {wy}, {wz})")
             return

        # 4. Set the voxel
        chunk.Voxels[voxel_index] = 1

        




    def GetTargetVoxel(self, ray_origin: np.ndarray, ray_direction: np.ndarray, max_distance: float = 10.0):
        """
        Performs Voxel Raycasting (DDA) to find the targeted block and placement position.

        Args:
            ray_origin: The world position of the camera (e.g., player position).
            ray_direction: The normalized direction vector the player is looking.
            max_distance: The maximum reach distance for the ray.

        Returns:
            A tuple: ((hit_x, hit_y, hit_z), (place_x, place_y, place_z))
            Returns (None, None) if no block is hit within max_distance.
        """
        
        # Ensure direction is normalized (important for DDA logic)
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        # 1. Initial Voxel Coordinates
        # Start at the integer coordinates of the origin
        step_x = math.copysign(1, ray_direction[0])
        step_y = math.copysign(1, ray_direction[1])
        step_z = math.copysign(1, ray_direction[2])
        
        # 2. Initial Grid Position
        # 'map_pos' tracks the integer coordinates of the current voxel being checked
        map_x = int(math.floor(ray_origin[0]))
        map_y = int(math.floor(ray_origin[1]))
        map_z = int(math.floor(ray_origin[2]))

        # 3. Ray Parameters and Deltas
        # 'tDelta' is the distance to travel along the ray to cross one unit in X/Y/Z.
        # It is calculated as 1 / |DirectionComponent|.
        tDelta_x = float('inf') if ray_direction[0] == 0 else abs(1.0 / ray_direction[0])
        tDelta_y = float('inf') if ray_direction[1] == 0 else abs(1.0 / ray_direction[1])
        tDelta_z = float('inf') if ray_direction[2] == 0 else abs(1.0 / ray_direction[2])

        # 4. Initial Ray Boundary Distances (tMax)
        # 'tMax' is the distance along the ray to the *first* voxel boundary in each direction.
        # 'fract' is the fractional part of the origin coordinate
        
        # Calculate initial tMax for X
        if ray_direction[0] < 0:
            tMax_x = (ray_origin[0] - map_x) * tDelta_x
        else:
            tMax_x = (map_x + 1.0 - ray_origin[0]) * tDelta_x
        
        # Calculate initial tMax for Y
        if ray_direction[1] < 0:
            tMax_y = (ray_origin[1] - map_y) * tDelta_y
        else:
            tMax_y = (map_y + 1.0 - ray_origin[1]) * tDelta_y

        # Calculate initial tMax for Z
        if ray_direction[2] < 0:
            tMax_z = (ray_origin[2] - map_z) * tDelta_z
        else:
            tMax_z = (map_z + 1.0 - ray_origin[2]) * tDelta_z

        # The current distance traveled along the ray
        current_t = 0.0

        # 5. Iterative Stepping (DDA Loop)
        while current_t < max_distance:
            # Determine the next boundary the ray will cross (minimum tMax)
            if tMax_x < tMax_y:
                if tMax_x < tMax_z:
                    # Step in X direction
                    current_t = tMax_x
                    tMax_x += tDelta_x
                    
                    # Update map position and hit normal
                    prev_x = map_x
                    map_x += int(step_x)
                    
                    # The normal points OUT of the empty block, so the placement 
                    # vector is the step taken to get to the new block.
                    # However, for the HIT block, the normal points away from the camera.
                    normal_x = -step_x  # e.g., if step_x is 1, normal is -1 (hit west face)
                    normal_y, normal_z = 0, 0
                    
                else:
                    # Step in Z direction
                    current_t = tMax_z
                    tMax_z += tDelta_z

                    prev_z = map_z
                    map_z += int(step_z)

                    normal_x, normal_y = 0, 0
                    normal_z = -step_z
                    
            else: # tMax_y < tMax_x
                if tMax_y < tMax_z:
                    # Step in Y direction
                    current_t = tMax_y
                    tMax_y += tDelta_y

                    prev_y = map_y
                    map_y += int(step_y)

                    normal_x, normal_z = 0, 0
                    normal_y = -step_y
                    
                else:
                    # Step in Z direction (or Y if tMax_y == tMax_z)
                    current_t = tMax_z
                    tMax_z += tDelta_z
                    
                    prev_z = map_z
                    map_z += int(step_z)

                    normal_x, normal_y = 0, 0
                    normal_z = -step_z
                    
            # 6. Check for Block Hit
            # Attempt to get the block type at the current map position (map_x, map_y, map_z)
            
            chunk_key = (map_x // Chunk_Size, map_y // Chunk_Size, map_z // Chunk_Size)
            
            if chunk_key in self.WorldMap:
                chunk = self.WorldMap[chunk_key]
                
                local_x = map_x % Chunk_Size
                local_y = map_y % Chunk_Size
                local_z = map_z % Chunk_Size
                voxel_index = local_x + local_y * Chunk_Size + local_z * Chunk_Size * Chunk_Size
                
                block_type = chunk.Voxels[voxel_index]
                
                if block_type != 0:
                    # Block Hit!
                    hit_pos = (map_x, map_y, map_z)
                    
                    # Placement position is the hit position plus the normal vector,
                    # which points OUT of the hit face and into the adjacent empty space.
                    place_pos = (map_x + normal_x, map_y + normal_y, map_z + normal_z)
                    
                    return hit_pos, place_pos
                    
        return None, None