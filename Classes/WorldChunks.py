from dataclasses import dataclass,field
from OpenGL.GL import *
from Classes.FrustumCull import FrustumCulling 
import numpy as np

Chunk_Size=8
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
            worldPos=chunk.Position*chunk.Size

            mins=worldPos
            maxs=worldPos+chunk.Size

            if frustum.aabb_visible(mins,maxs):
                visible.append((pos,chunk))
        return visible
            
            
    def DrawVisiChunks(self,frustum, instanceVBO_ID, EBO_ID):
        visibleChunks=self.GetVisiChunks(frustum)

        maxCubes=len(visibleChunks)*Voxel_Count

        if maxCubes==0:
            return

        tempPositions=np.zeros((maxCubes,3),dtype=np.float32)
        tempBlockTypes=np.zeros(maxCubes,dtype=np.float32)


        count=0
        # ️ Collect positions and block types
        for key, chunk in visibleChunks:
            chunk_pos_int= chunk.Position.astype(np.int32) 
            chunk_offset = chunk_pos_int * Chunk_Size

            for z in range(Chunk_Size):
                for y in range(Chunk_Size):
                    for x in range(Chunk_Size):

                        if voxel.BlockType==0 : continue

                        #Global position of the voxel
                        vx=chunk_offset[0]+x
                        vy=chunk_offset[1]+y
                        vz=chunk_offset[2]+z

                        #--- Face Culling starts ---
                        # We need the VBO data for a single face (6 indices, 4 vertices)
                        # You will create a temporary array to hold the vertices/indices for the visible faces.

                        # Check the Z+ face (Forward)
                        if not self.is_solid_at_world_coords(vx, vy, vz + 1): 
                            # Add vertices and indices for the Z+ (Forward) face to the instance data
                            #self.add_face_to_mesh(Face.FRONT, vx, vy, vz, count)
                            count += 1

                        
                        index = x + y * Chunk_Size + z * Chunk_Size * Chunk_Size
                        voxel = chunk.Voxels[index]

                        tempPositions[count,0]=chunk_offset[0]+x
                        tempPositions[count,1]=chunk_offset[1]+y
                        tempPositions[count,2]=chunk_offset[2]+z

                        tempBlockTypes[count]=voxel.BlockType

                        count+=1

        self.VisibleCubePositions = tempPositions[:count]
        blockTypes = tempBlockTypes[:count]

        #  Upload instance data to GPU (after loops)
        if count>0:
            instance_data = np.zeros((len(self.VisibleCubePositions), 4), dtype=np.float32)
            instance_data[:, :3] = self.VisibleCubePositions
            instance_data[:, 3] = blockTypes

            glBindBuffer(GL_ARRAY_BUFFER, instanceVBO_ID)

            #optimization called orphaning the buffer
            glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, None, GL_DYNAMIC_DRAW)
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