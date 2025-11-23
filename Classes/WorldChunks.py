from dataclasses import dataclass,field
from OpenGL.GL import *
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
                        chunk.Voxels[i].BlockType=1
                    elif cz>0:
                        chunk.Voxels[i].BlockType=2
                    else:
                        chunk.Voxels[i].BlockType=3



                #Insert into Hashmap(dict)
                key=(cx,0,cz)
                self.WorldMap[key]=chunk
            
    def DrawVisiChunks(self, instanceVBO_ID, EBO_ID):
        self.VisibleCubePositions.clear()
        blockTypes = []

        # 1️⃣ Collect positions and block types
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
                        blockTypes.append(voxel.BlockType)

        # 2️⃣ Upload instance data to GPU (after loops)
        if self.VisibleCubePositions:
            instance_data = np.zeros((len(self.VisibleCubePositions), 4), dtype=np.float32)
            instance_data[:, :3] = self.VisibleCubePositions
            instance_data[:, 3] = blockTypes

            glBindBuffer(GL_ARRAY_BUFFER, instanceVBO_ID)
            glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)

            # 3️⃣ Draw instanced
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_ID)
            glDrawElementsInstanced(
                GL_TRIANGLES,
                36,  # number of indices per cube
                GL_UNSIGNED_INT,
                None,
                len(self.VisibleCubePositions)
            )