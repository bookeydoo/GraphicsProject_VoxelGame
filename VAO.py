from OpenGL.GL import *
import ctypes


class VAO:
    def __init__(self):
        self.ID=glGenVertexArrays(1)
    
    def LinkVBO(self,VBO,layout,numComp,typeEnum,stride,offset):
        VBO.bind()
        
        offsetPtr=ctypes.c_void_p(offset)

        glVertexAttribPointer(
            layout,
            numComp,
            typeEnum,
            GL_FALSE,
            stride,
            offsetPtr
        )
        glEnableVertexAttribArray(layout)
        VBO.unbind()

    def bind(self):
        glBindVertexArray(self.ID)
    
    def unbind(self):
        glBindVertexArray(0)
    
    def delete(self):
        glDeleteVertexArrays(1,[self.ID])