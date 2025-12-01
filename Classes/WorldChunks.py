from dataclasses import dataclass,field
from OpenGL.GL import *
from Classes.FrustumCull import FrustumCulling 
import numpy as np
from PIL import Image

Chunk_Size=4
Voxel_Count=Chunk_Size**3

@dataclass
class Voxel:
    BlockType: np.uint32=np.uint32(0)


@dataclass
class Chunk:
    Position:  np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    Size: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    Voxels: list = field(default_factory=lambda:[Voxel() for _ in range(Voxel_Count)])

class World:

    def __init__(self):
        self.WorldMap: dict[tuple,Chunk]={}
        self.VisibleCubePositions: list[tuple]=[]
        self.noise_array = None  # Store the noise texture data
        
        self.FACE_DATA = {
            'RIGHT': (
                np.array([
                    # POS(X Y Z) ,tex ( U V ),center(x z) and blocktype
                    0.5, -0.5, 0.5,  0.0, 0.0, 0.0, 0.0, 0.0,
                    0.5,  0.5, 0.5,  1.0, 0.0, 0.0, 0.0, 1.0,
                    0.5,  0.5, -0.5, 1.0, 1.0, 1.0, 1.0, 1.0,
                    0.5, -0.5, -0.5, 0.0, 1.0, 1.0, 1.0, 0.0,
                ], dtype=np.float32),
                np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
            ),
            'LEFT': (
                np.array([
                    -0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0,
                    -0.5,  0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 1.0,
                    -0.5,  0.5, -0.5,1.0, 1.0, 0.0, 0.0, 1.0,
                    -0.5, -0.5, -0.5,0.0, 1.0, 1.0, 0.0, 0.0,
                ], dtype=np.float32),
                np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
            ),
            'TOP': (
                np.array([
                    -0.5, 0.5, 0.5,   0.0, 0.0, 0.0, 0.0, 0.0,
                    0.5, 0.5, 0.5,    1.0, 0.0, 1.0, 1.0, 0.0,
                    -0.5, 0.5, -0.5,  1.0, 1.0, 0.0, 0.0, 1.0,
                    0.5, 0.5, -0.5,   0.0, 1.0, 1.0, 1.0, 1.0,
                ], dtype=np.float32),
                np.array([0, 2, 3, 0, 1, 3], dtype=np.uint32)
            ),
            'BOTTOM': (
                np.array([
                    -0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0,
                    0.5, -0.5, -0.5, 1.0, 0.0, 1.0, 1.0, 1.0,
                    -0.5, -0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0,
                    0.5, -0.5, -0.5, 0.0, 1.0, 1.0, 0.0, 1.0,
                ], dtype=np.float32),
                np.array([0, 2, 1, 1, 3, 2], dtype=np.uint32)
            ),
            'FRONT': (
                np.array([
                    -0.5, -0.5, 0.5,0.0, 0.0, 0.0, 0.0, 0.0,
                    -0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0,
                    0.5, -0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.0,
                    0.5, 0.5, 0.5,  0.0, 1.0, 0.0, 1.0, 1.0,
                ], dtype=np.float32),
                np.array([0, 1, 2, 2, 3, 1], dtype=np.uint32)
            ),
            'BACK': (
                np.array([
                    -0.5, -0.5, -0.5,0.0, 0.0, 0.0, 0.0, 0.0,
                    -0.5, 0.5, -0.5, 1.0, 0.0, 1.0, 0.0, 1.0,
                    0.5, 0.5, -0.5,  1.0, 1.0, 0.0, 1.0, 1.0,
                    0.5, -0.5, -0.5, 0.0, 1.0, 1.0, 0.0, 1.0,
                ], dtype=np.float32),
                np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
            )
        }

        self.Stride=8

    def load_noise_texture(self, path):
        """Load the Perlin noise texture for CPU-side terrain generation"""
        image = Image.open(path)
        self.noise_array = np.array(image.convert("L"), dtype=np.float32) / 255.0
        print(f"Loaded noise texture: {self.noise_array.shape}")

    def sample_noise(self, x, z):
        """Sample the noise texture at world coordinates x, z"""
        if self.noise_array is None:
            return 0.5  # Default if no texture loaded
        
        # Same UV calculation as in shader
        u = x * 0.05
        v = z * 0.05
        
        # Wrap coordinates (simulating GL_REPEAT)
        u = u % 1.0
        v = v % 1.0
        
        # Convert to pixel coordinates
        h, w = self.noise_array.shape
        px = int(u * w) % w
        py = int(v * h) % h
        
        return self.noise_array[py, px]

    def get_height_at(self, x, z):
        """Calculate terrain height at world position (x, z)"""
        noiseVal = self.sample_noise(x + 0.5, z + 0.5)  # Sample at voxel center
        
        # Same height calculation as shader
        heightScale = 2.0
        if noiseVal > 0.7:
            displacement = noiseVal * heightScale * 2.0
        elif noiseVal > 0.3:
            displacement = noiseVal * heightScale
        else:
            displacement = noiseVal * heightScale * 0.1
        
        height = int(displacement * 5.0)  # Convert to voxel units
        return max(0, height)  # Ensure non-negative

    def add_face_to_mesh(self,faceKey,wx,wy,wz,BlockType,vertex_data_list,index_data_list,vertex_count):
        face_vertices_flat, face_indices_rel = self.FACE_DATA[faceKey]
        face_vertices = face_vertices_flat.reshape((-1, self.Stride)).copy()
        
        # Offset positions
        face_vertices[:, 0] += wx
        face_vertices[:, 1] += wy
        face_vertices[:, 2] += wz
        
        # Set voxel center and block type
        face_vertices[:, 5] = wx + 0.5  # voxel center x 
        face_vertices[:, 6] = wz + 0.5  # voxel center z 
        face_vertices[:, 7] = float(BlockType)  # Block type 

        vertex_data_list.append(face_vertices.flatten('C'))
        face_indices_abs = face_indices_rel + vertex_count 
        index_data_list.append(face_indices_abs)
        
        return len(face_vertices)

    def is_solid_at_world_coords(self, wx, wy, wz):
        chunk_x = int(wx // Chunk_Size)
        chunk_y = int(wy // Chunk_Size)
        chunk_z = int(wz // Chunk_Size)
        
        chunk_key = (chunk_x, chunk_y, chunk_z)

        if chunk_key not in self.WorldMap:
            return False

        chunk = self.WorldMap[chunk_key]

        voxel_x = int(wx % Chunk_Size)
        voxel_y = int(wy % Chunk_Size)
        voxel_z = int(wz % Chunk_Size)
        
        if voxel_x < 0: voxel_x += Chunk_Size
        if voxel_y < 0: voxel_y += Chunk_Size
        if voxel_z < 0: voxel_z += Chunk_Size

        index = voxel_x + voxel_y * Chunk_Size + voxel_z * Chunk_Size * Chunk_Size
        
        return chunk.Voxels[index].BlockType > 0
        
    def InitWorld(self, PlayerPos, noise_texture_path="Resources/perlin_noise.png"):
        """Initialize world with terrain generation based on Perlin noise"""
        
        # Load noise texture
        import os
        if os.path.exists(noise_texture_path):
            self.load_noise_texture(noise_texture_path)
        else:
            print(f"Warning: Could not find noise texture at {noise_texture_path}")
        
        radius = 4
        max_height = 10  # Maximum terrain height in voxels

        for cx in range(-radius, radius + 1):
            for cz in range(-radius, radius + 1):
                # Generate chunks vertically based on terrain height in this XZ column
                for cy in range(0, max_height // Chunk_Size + 1):
                    chunk = Chunk()
                    chunk.Position = np.array([cx, cy, cz], dtype=np.float32)
                    chunk.Size = np.array([Chunk_Size] * 3, dtype=np.float32)

                    # Fill voxels based on terrain height
                    for z in range(Chunk_Size):
                        for x in range(Chunk_Size):
                            # World coordinates
                            world_x = cx * Chunk_Size + x
                            world_z = cz * Chunk_Size + z
                            
                            # Get terrain height at this XZ position
                            terrain_height = self.get_height_at(world_x, world_z)
                            
                            # Fill voxels vertically up to terrain height
                            for y in range(Chunk_Size):
                                world_y = cy * Chunk_Size + y
                                index = x + y * Chunk_Size + z * Chunk_Size * Chunk_Size
                                
                                if world_y <= terrain_height:
                                    # Assign block type based on height or XZ position
                                    if world_y == terrain_height:
                                        chunk.Voxels[index].BlockType = 1  # Grass/top layer
                                    elif world_y > terrain_height - 3:
                                        chunk.Voxels[index].BlockType = 2  # Dirt
                                    else:
                                        chunk.Voxels[index].BlockType = 3  # Stone
                                else:
                                    chunk.Voxels[index].BlockType = 0  # Air

                    # Insert chunk into world map
                    key = (cx, cy, cz)
                    self.WorldMap[key] = chunk

        print(f"Generated {len(self.WorldMap)} chunks")
    
    def GetVisiChunks(self,frustum):
        visible=[]
        for pos,chunk in self.WorldMap.items():
            worldPos=chunk.Position * Chunk_Size

            mins=worldPos
            maxs=worldPos+ np.array([Chunk_Size,Chunk_Size,Chunk_Size],dtype=np.float32)

            if frustum.aabb_visible(mins,maxs):
                visible.append((pos,chunk))
        return visible
            
    def DrawVisiChunks(self, frustum, CubeVBO_ID, EBO_ID, VAO_ID):
        visibleChunks = self.GetVisiChunks(frustum)

        total_vertex_data = [] 
        total_index_data = []
        vertex_count = 0

        def is_solid(cx, cy, cz):
            return self.is_solid_at_world_coords(cx, cy, cz)

        for key, chunk in visibleChunks:
            chunk_pos_int = chunk.Position.astype(np.int32) 
            chunk_offset = chunk_pos_int * Chunk_Size
            
            for z in range(Chunk_Size):
                for y in range(Chunk_Size):
                    for x in range(Chunk_Size):
                        index = x + y * Chunk_Size + z * Chunk_Size * Chunk_Size
                        voxel = chunk.Voxels[index]
                        block_type = voxel.BlockType

                        if block_type == 0: continue

                        wx = chunk_offset[0] + x
                        wy = chunk_offset[1] + y
                        wz = chunk_offset[2] + z
                        
                        # Check all 6 faces for visibility
                        if not is_solid(wx + 1, wy, wz):
                            face_vertices = self.add_face_to_mesh('RIGHT', wx, wy, wz, block_type, total_vertex_data, total_index_data, vertex_count)
                            vertex_count += face_vertices
                        
                        if not is_solid(wx - 1, wy, wz):
                            face_vertices = self.add_face_to_mesh('LEFT', wx, wy, wz, block_type, total_vertex_data, total_index_data, vertex_count)
                            vertex_count += face_vertices

                        if not is_solid(wx, wy + 1, wz):
                            face_vertices = self.add_face_to_mesh('TOP', wx, wy, wz, block_type, total_vertex_data, total_index_data, vertex_count)
                            vertex_count += face_vertices

                        if not is_solid(wx, wy - 1, wz):
                            face_vertices = self.add_face_to_mesh('BOTTOM', wx, wy, wz, block_type, total_vertex_data, total_index_data, vertex_count)
                            vertex_count += face_vertices

                        if not is_solid(wx, wy, wz + 1):
                            face_vertices = self.add_face_to_mesh('FRONT', wx, wy, wz, block_type, total_vertex_data, total_index_data, vertex_count)
                            vertex_count += face_vertices
                            
                        if not is_solid(wx, wy, wz - 1):
                            face_vertices = self.add_face_to_mesh('BACK', wx, wy, wz, block_type, total_vertex_data, total_index_data, vertex_count)
                            vertex_count += face_vertices

        if len(total_index_data) > 0:
            final_vertices = np.concatenate(total_vertex_data).astype(np.float32)
            final_indices = np.concatenate(total_index_data).astype(np.uint32)
            total_indices = len(final_indices)


            glBindBuffer(GL_ARRAY_BUFFER, CubeVBO_ID) 
            glBufferData(GL_ARRAY_BUFFER, final_vertices.nbytes, final_vertices, GL_DYNAMIC_DRAW) 
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_ID)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, final_indices.nbytes, final_indices, GL_DYNAMIC_DRAW)

            glBindVertexArray(VAO_ID)
            glDrawElements(GL_TRIANGLES, total_indices, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)