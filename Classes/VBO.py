
from OpenGL.GL import *

class VBO:
    def __init__(self,vertices=None,Static=True):
        self.ID=glGenBuffers(1)
        if vertices is not None and Static:
            self.bind()
            glBufferData(GL_ARRAY_BUFFER,vertices.nbytes,vertices,GL_STATIC_DRAW)
        else:
            self.bind()


    def bind(self):
        glBindBuffer(GL_ARRAY_BUFFER,self.ID)
    
    def unbind(self):
        glBindBuffer(GL_ARRAY_BUFFER,0)
    
    def delete(self):
        glDeleteBuffers(1,[self.ID])