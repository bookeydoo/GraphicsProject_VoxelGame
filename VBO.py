
from OpenGL.GL import *

class VBO:
    def __init__(self,vertices=None):
        self.ID=glGenBuffers(1)
        if vertices is not None:
            self.bind()
            glBufferData(GL_ARRAY_BUFFER,vertices.nbytes,vertices,GL_STATIC_DRAW)


    def bind(self):
        glBindBuffer(GL_ARRAY_BUFFER,self.ID)
    
    def unbind(self):
        glBindBuffer(GL_ARRAY_BUFFER,0)
    
    def delete(self):
        glDeleteVertexArrays(1,[self.ID])