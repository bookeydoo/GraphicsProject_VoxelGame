from dataclasses import dataclass,field
from OpenGL.GL import *
from Classes.FrustumCull import FrustumCulling 
import numpy as np

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

	#std::unordered_map<glm::vec3, Chunk*> worldMap;
	#std::vector<glm::vec3> visibleCubePositions;
    def __init__(self):
        self.WorldMap: dict[tuple,Chunk]={}
        self.VisibleCubePositions: list[tuple]=[]
        self.FACE_DATA = {
            # +X (Right) Face: Indices 16-19 in your main.py vertices array
            # We only need the 4 vertices' data (32 floats) and 6 indices
            'RIGHT': (
                np.array([
                    0.5, -0.5, 0.5,  1.0, 1.0, 0.0, 0.0, 0.0,
                    0.5,  0.5, 0.5,  1.0, 0.4, 0.0, 0.0, 1.0,
                    0.5,  0.5, -0.5, 1.0, 0.2, 1.0, 1.0, 1.0,
                    0.5, -0.5, -0.5, 1.0, 0.3, 1.0, 1.0, 0.0,
                ], dtype=np.float32),
                np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32) # Indices for these 4 vertices
            ),
            # -X (Left) Face: Indices 20-23
            'LEFT': (
                np.array([
                    -0.5, -0.5, 0.5, 1.0, 0.2, 1.0, 1.0, 0.0,
                    -0.5,  0.5, 0.5, 1.0, 0.4, 0.0, 1.0, 1.0,
                    -0.5,  0.5, -0.5, 1.0, 1.0, 0.0, 0.0, 1.0,
                    -0.5, -0.5, -0.5, 1.0, 0.3, 1.0, 0.0, 0.0,
                ], dtype=np.float32),
                np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
            ),
            # +Y (Top) Face: Indices 8-11
            'TOP': (
                np.array([
                    -0.5, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0,
                    0.5, 0.5, 0.5, 1.0, 0.2, 1.0, 1.0, 0.0,
                    -0.5, 0.5, -0.5, 1.0, 0.4, 0.0, 0.0, 1.0,
                    0.5, 0.5, -0.5, 1.0, 0.3, 1.0, 1.0, 1.0,
                ], dtype=np.float32),
                np.array([0, 2, 3, 0, 1, 3], dtype=np.uint32) # Corrected indices for top face geometry
            ),
            # -Y (Bottom) Face: Indices 12-15 (Note: Your main.py bottom indices are strange, using a correct template)
            'BOTTOM': (
                np.array([
                    -0.5, -0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.0,
                    0.5, -0.5, -0.5, 1.0, 0.2, 1.0, 1.0, 1.0,
                    -0.5, -0.5, 0.5, 1.0, 0.4, 0.0, 0.0, 0.0,
                    0.5, -0.5, -0.5, 1.0, 0.3, 1.0, 0.0, 1.0,
                ], dtype=np.float32),
                np.array([0, 2, 1, 1, 3, 2], dtype=np.uint32)
            ),
            # +Z (Front) Face: Indices 0-3
            'FRONT': (
                np.array([
                    -0.5, -0.5, 0.5, 0.5, 0.2, 0.0, 0.0, 0.0,
                    -0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 1.0,
                    0.5, -0.5, 0.5, 0.2, 0.3, 1.0, 1.0, 0.0,
                    0.5, 0.5, 0.5, 1.0, 1.0, 0.0, 1.0, 1.0,
                ], dtype=np.float32),
                np.array([0, 1, 2, 2, 3, 1], dtype=np.uint32)
            ),
            # -Z (Back) Face: Indices 4-7
            'BACK': (
                np.array([
                    -0.5, -0.5, -0.5, 1.0, 1.0, 0.0, 0.0, 0.0,
                    -0.5, 0.5, -0.5, 1.0, 0.2, 1.0, 0.0, 1.0,
                    0.5, 0.5, -0.5, 1.0, 0.4, 0.0, 1.0, 1.0,
                    0.5, -0.5, -0.5, 1.0, 0.3, 1.0, 0.0, 1.0,
                ], dtype=np.float32),
                np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
            )
        }

        self.Stride=8

    
    def add_face_to_mesh(self,faceKey,wx,wy,wz,BlockType,vertex_data_list,index_data_list,vertex_count):
        """
        Calculates and appends the vertices and indices for one visible face.
        
        Args:
            face_key (str): 'TOP', 'BOTTOM', 'FRONT', etc.
            wx, wy, wz (float): World position of the voxel (min corner).
            block_type (int): The BlockType ID to assign to the instance data.
            vertex_data_list (list): The list to append vertex data to.
            index_data_list (list): The list to append index data to.
            vertex_count (int): The current total number of vertices in the mesh.
        
        Returns:
            int: The new total number of indices added.
        """
        # 1. Get base vertex and index data for the face
        face_vertices_flat, face_indices_rel = self.FACE_DATA[faceKey]
        
        # We have 4 vertices, 8 floats each (pos[3], color[3], tex[2])
        face_vertices = face_vertices_flat.reshape((-1, self.Stride)).copy()
        
        # 2. Offset the positions
        # The first 3 components are positions (x, y, z)
        # We shift all 4 vertices by the voxel's world position (wx, wy, wz)
        face_vertices[:, 0] += wx
        face_vertices[:, 1] += wy
        face_vertices[:, 2] += wz
        
        # NEW: Inject BlockType ID (1-4) into the R component (index 3)
        # This corresponds to the 'color' attribute (Layout 1) in your VAO link.
        face_vertices[:, 3] = BlockType  # R component (BlockType ID)
        face_vertices[:, 4] = 0.0        # G component (set to 0)
        face_vertices[:, 5] = 0.0        # B component (set to 0)




        # 3. Append to main lists
        # Vertices (flat)
        vertex_data_list.append(face_vertices.flatten())
        
        # Indices (relative to the global vertex count)
        # The 4 vertices we just added start at index 'vertex_count'
        face_indices_abs = face_indices_rel + vertex_count 
        index_data_list.append(face_indices_abs)
        
        # 4. Return new vertex count
        return len(face_vertices)


    
    def is_solid_at_world_coords(self, wx, wy, wz):
        # 1. Determine the CHUNK containing the world coordinates
        # Chunk coordinates (integer division)
        chunk_x = int(wx // Chunk_Size)
        chunk_y = int(wy // Chunk_Size)
        chunk_z = int(wz // Chunk_Size)
        
        chunk_key = (chunk_x, chunk_y, chunk_z)

        # 2. Check if the chunk exists (i.e., if it's outside the generated radius)
        if chunk_key not in self.WorldMap:
            # If the chunk doesn't exist, treat the space as air (BlockType 0)
            return False

        chunk = self.WorldMap[chunk_key]

        # 3. Determine the VOXEL index within the chunk
        # Voxel coordinates (modulo operation)
        voxel_x = int(wx % Chunk_Size)
        voxel_y = int(wy % Chunk_Size)
        voxel_z = int(wz % Chunk_Size)
        
        # Handle negative coordinates (Python's % handles this correctly for positive Chunk_Size)
        if voxel_x < 0: voxel_x += Chunk_Size
        if voxel_y < 0: voxel_y += Chunk_Size
        if voxel_z < 0: voxel_z += Chunk_Size

        # Voxel index calculation
        index = voxel_x + voxel_y * Chunk_Size + voxel_z * Chunk_Size * Chunk_Size
        
        # Check if the neighbor block is solid
        return chunk.Voxels[index].BlockType > 0
        
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
                    if cz<0 and cx>4:
                        chunk.Voxels[i].BlockType=1
                    elif cz>0 and cx<4:
                        chunk.Voxels[i].BlockType=2
                    elif cz>0 and cx>4:
                        chunk.Voxels[i].BlockType=3
                    else :
                        chunk.Voxels[i].BlockType=0



                #Insert into Hashmap(dict)
                key=(cx,0,cz)
                self.WorldMap[key]=chunk
    
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
        # We no longer use instanceVBO_ID and EBO_ID for the Instancing method.
        # We need a new VBO and EBO for the merged mesh, or reuse the existing ones
        # with different data. For now, we'll assume the mesh data goes into the VBO
        # you originally defined for the cube shape, and the EBO you originally defined.
        
        visibleChunks = self.GetVisiChunks(frustum)

        # ----------------------------------------------------
        # 1. Data Structures for the Merged Mesh
        # ----------------------------------------------------
        
        # Use Python lists to dynamically collect data (faster than growing numpy array)
        total_vertex_data = [] 
        total_index_data = []
        vertex_count = 0

        # The neighbor checking logic
        def is_solid(cx, cy, cz):
            # This is the full implementation of the helper function needed for culling
            return self.is_solid_at_world_coords(cx, cy, cz)

        # ----------------------------------------------------
        # 2. Iterate and Cull Faces
        # ----------------------------------------------------
        for key, chunk in visibleChunks:
            # Chunk offset in world space
            chunk_pos_int = chunk.Position.astype(np.int32) 
            chunk_offset = chunk_pos_int * Chunk_Size
            
            for z in range(Chunk_Size):
                for y in range(Chunk_Size):
                    for x in range(Chunk_Size):
                        index = x + y * Chunk_Size + z * Chunk_Size * Chunk_Size
                        voxel = chunk.Voxels[index]
                        block_type = voxel.BlockType

                        if block_type == 0: continue # Skip air blocks

                        # World coordinates of the current voxel's minimum corner
                        wx = chunk_offset[0] + x
                        wy = chunk_offset[1] + y
                        wz = chunk_offset[2] + z
                        
                        # Check all 6 faces for visibility
                        
                        # +X (Right)
                        if not is_solid(wx + 1, wy, wz):
                            face_vertices = self.add_face_to_mesh('RIGHT', wx, wy, wz, block_type, total_vertex_data, total_index_data, vertex_count)
                            vertex_count += face_vertices
                        
                        # -X (Left)
                        if not is_solid(wx - 1, wy, wz):
                            face_vertices = self.add_face_to_mesh('LEFT', wx, wy, wz, block_type, total_vertex_data, total_index_data, vertex_count)
                            vertex_count += face_vertices

                        # +Y (Top)
                        if not is_solid(wx, wy + 1, wz):
                            face_vertices = self.add_face_to_mesh('TOP', wx, wy, wz, block_type, total_vertex_data, total_index_data, vertex_count)
                            vertex_count += face_vertices

                        # -Y (Bottom)
                        if not is_solid(wx, wy - 1, wz):
                            face_vertices = self.add_face_to_mesh('BOTTOM', wx, wy, wz, block_type, total_vertex_data, total_index_data, vertex_count)
                            vertex_count += face_vertices

                        # +Z (Front)
                        if not is_solid(wx, wy, wz + 1):
                            face_vertices = self.add_face_to_mesh('FRONT', wx, wy, wz, block_type, total_vertex_data, total_index_data, vertex_count)
                            vertex_count += face_vertices
                            
                        # -Z (Back)
                        if not is_solid(wx, wy, wz - 1):
                            face_vertices = self.add_face_to_mesh('BACK', wx, wy, wz, block_type, total_vertex_data, total_index_data, vertex_count)
                            vertex_count += face_vertices


        # ----------------------------------------------------
        # 3. Finalize Data and Upload to GPU
        # ----------------------------------------------------
        total_indices = len(total_index_data) * 6
        
        if len(total_index_data) > 0:
            final_vertices = np.concatenate(total_vertex_data).astype(np.float32)
            final_indices = np.concatenate(total_index_data).astype(np.uint32)

            total_indices=len(final_indices)

            # --- A. BIND VAO (The Fix for flickering) ---
            glBindVertexArray(VAO_ID)

            # --- B. Update Vertex VBO (CubeVBO) ---
            # NOTE: We assume 'instanceVBO_ID' is the ID of your CubeVBO (VBO for geometry)
            glBindBuffer(GL_ARRAY_BUFFER,CubeVBO_ID) 
            glBufferData(GL_ARRAY_BUFFER, final_vertices.nbytes, final_vertices, GL_DYNAMIC_DRAW) 

            # --- C. Update Index EBO ---
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_ID)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, final_indices.nbytes, final_indices, GL_DYNAMIC_DRAW)

            # --- D. Draw Merged Mesh ---
            glDrawElements(
                GL_TRIANGLES,
                total_indices,  # Total number of indices to draw
                GL_UNSIGNED_INT,
                None,
            )

            # --- E. Clean State ---
            glBindVertexArray(0)