from dataclasses import dataclass, field
from OpenGL.GL import *
import numpy as np
import math

Chunk_Size = 2
Voxel_Count = Chunk_Size**3

@dataclass
class Voxel:
    BlockType: np.uint32 = np.uint32(0)

@dataclass
class Chunk:
    Position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    Size: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    Voxels: np.ndarray = field(default_factory=lambda: np.zeros(Voxel_Count, dtype=np.uint32))
    Colors: np.ndarray = field(default_factory=lambda: np.ones((Voxel_Count, 4), dtype=np.float32))

class World:
    def __init__(self):
        self.WorldMap: dict[tuple, Chunk] = {}
        self.VisibleCubePositions: list[tuple] = []
        self.BlockTypes = []
      
    def InitWorld(self, PlayerPos):
        radius = 2
        y_min = -1
        y_max = 3

        for cx in range(-radius, radius+1):
            for cy in range(y_min, y_max):
                for cz in range(-radius, radius+1):
                    self._CreateChunk(cx, cy, cz)
    
    def _CreateChunk(self, cx, cy, cz):
        """Internal method to create a chunk at given coordinates"""
        chunk = Chunk()
        chunk.Position = np.array([cx, cy, cz], dtype=np.float32)
        chunk.Size = np.array([Chunk_Size]*3, dtype=np.float32)

        # Initialize voxels based on position (your perlin noise logic goes here)
        chunk.Voxels=np.zeros(Voxel_Count,dtype=np.uint32)

        key = (cx, cy, cz)
        self.WorldMap[key] = chunk
        return chunk
    
    def GetOrCreateChunk(self, chunk_coords):
        """Get a chunk, creating it if it doesn't exist"""
        if chunk_coords not in self.WorldMap:
            cx, cy, cz = chunk_coords
            return self._CreateChunk(cx, cy, cz)
        return self.WorldMap[chunk_coords]
            
    def DrawVisiChunks(self, instanceVBO_ID, EBO_ID):
        self.VisibleCubePositions.clear()
        self.BlockTypes.clear()
        self.BlockColors=[]

        # Collect positions and block types
        for key, chunk in self.WorldMap.items():
            chunk_offset = chunk.Position * Chunk_Size

            for z in range(Chunk_Size):
                for y in range(Chunk_Size):
                    for x in range(Chunk_Size):
                        index = x + y * Chunk_Size + z * Chunk_Size * Chunk_Size
                        voxel = chunk.Voxels[index]

                        if voxel == 0:
                            continue

                        local_pos = np.array([x, y, z], dtype=np.float32)
                        world_pos = chunk_offset + local_pos

                        self.VisibleCubePositions.append(world_pos)
                        self.BlockTypes.append(chunk.Voxels[index])
                        self.BlockColors.append(chunk.Colors[index]) # Get the saved color

        # Upload instance data to GPU
        if self.VisibleCubePositions:
            instance_data = np.zeros((len(self.VisibleCubePositions), 8), dtype=np.float32)
            instance_data[:, :3] = self.VisibleCubePositions
            instance_data[:, 3] = np.array(self.BlockTypes, dtype=np.float32) 
            instance_data[:, 4:] = self.BlockColors # Add RGBA to the buffer

            glBindBuffer(GL_ARRAY_BUFFER, instanceVBO_ID)
            glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_ID)
            glDrawElementsInstanced(
                GL_TRIANGLES,
                36,
                GL_UNSIGNED_INT,
                None,
                len(self.VisibleCubePositions)
            )
    
    def DrawVoxel(self, worldPos: np.ndarray, BlockType,BlockColor):
        """Place a block at world position, creating chunks as needed"""
        
        wx, wy, wz = map(int, np.floor(worldPos))
        

        # Compute chunk coordinates
        cx, cy, cz = wx // Chunk_Size, wy // Chunk_Size, wz // Chunk_Size
        chunk_coords = (cx, cy, cz)

        # Get or create the chunk
        chunk = self.GetOrCreateChunk(chunk_coords)

        # Compute local voxel coordinates
        lx = wx % Chunk_Size
        ly = wy % Chunk_Size
        lz = wz % Chunk_Size

        # Compute 1D voxel index
        voxel_index = lx + ly * Chunk_Size + lz * Chunk_Size * Chunk_Size

        if not (0 <= voxel_index < Voxel_Count):
            print(f"Index {voxel_index} out of bounds!")
            return

      
        print(f"Current voxel value: {chunk.Voxels[voxel_index]}")
        chunk.Voxels[voxel_index] = BlockType
        chunk.Colors[voxel_index] = BlockColor

    def RemoveVoxel(self, worldPos: np.ndarray):
        """Remove a block at world position"""
        wx, wy, wz = map(int, np.floor(worldPos))
        
        cx, cy, cz = wx // Chunk_Size, wy // Chunk_Size, wz // Chunk_Size
        chunk_coords = (cx, cy, cz)

        if chunk_coords not in self.WorldMap:
            return

        chunk = self.WorldMap[chunk_coords]

        lx = wx % Chunk_Size
        ly = wy % Chunk_Size
        lz = wz % Chunk_Size

        voxel_index = lx + ly * Chunk_Size + lz * Chunk_Size * Chunk_Size

        if 0 <= voxel_index < Voxel_Count:
            chunk.Voxels[voxel_index] = 0
            print(f"Block removed at ({wx}, {wy}, {wz})")

    def GetTargetVoxel(self, ray_origin: np.ndarray, ray_direction: np.ndarray, max_distance: float = 10.0):

        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        x, y, z = map(int, np.floor(ray_origin))

        step_x = int(np.sign(ray_direction[0]))
        step_y = int(np.sign(ray_direction[1]))
        step_z = int(np.sign(ray_direction[2]))

        tMax_x = ((x + (step_x > 0)) - ray_origin[0]) / ray_direction[0] if ray_direction[0] != 0 else float('inf')
        tMax_y = ((y + (step_y > 0)) - ray_origin[1]) / ray_direction[1] if ray_direction[1] != 0 else float('inf')
        tMax_z = ((z + (step_z > 0)) - ray_origin[2]) / ray_direction[2] if ray_direction[2] != 0 else float('inf')

        tDelta_x = abs(1 / ray_direction[0]) if ray_direction[0] != 0 else float('inf')
        tDelta_y = abs(1 / ray_direction[1]) if ray_direction[1] != 0 else float('inf')
        tDelta_z = abs(1 / ray_direction[2]) if ray_direction[2] != 0 else float('inf')

        distance_traveled = 0.0
        hit_normal = (0, 0, 0)
        # Keep track of last empty voxel along the ray
        last_empty = (x, y, z)

        while distance_traveled <= max_distance:

            # Step to next voxel
            if tMax_x < tMax_y:
                if tMax_x < tMax_z:
                    x += step_x
                    hit_normal = (-step_x, 0, 0)
                    distance_traveled = tMax_x
                    tMax_x += tDelta_x
                else:
                    z += step_z
                    hit_normal = (0, 0, -step_z)
                    distance_traveled = tMax_z
                    tMax_z += tDelta_z
            else:
                if tMax_y < tMax_z:
                    y += step_y
                    hit_normal = (0, -step_y, 0)
                    distance_traveled = tMax_y
                    tMax_y += tDelta_y
                else:
                    z += step_z
                    hit_normal = (0, 0, -step_z)
                    distance_traveled = tMax_z
                    tMax_z += tDelta_z

            last_empty=(x,y,z)

            chunk_key = (x // Chunk_Size, y // Chunk_Size, z // Chunk_Size)
            if chunk_key not in self.WorldMap:
                continue

            chunk = self.WorldMap[chunk_key]
            lx = x - chunk_key[0] * Chunk_Size
            ly = y - chunk_key[1] * Chunk_Size
            lz = z - chunk_key[2] * Chunk_Size

            voxel_index = lx + ly * Chunk_Size + lz * Chunk_Size * Chunk_Size
            if 0 <= voxel_index < Voxel_Count:
                if chunk.Voxels[voxel_index] != 0:
                    hit_pos = (x, y, z)
                    place_pos = (
                        x + hit_normal[0],
                        y + hit_normal[1],
                        z + hit_normal[2]
                    )
                    return hit_pos, place_pos

        return None, last_empty 