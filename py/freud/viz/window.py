from __future__ import division, print_function
import numpy
import math
from ctypes import c_void_p
import OpenGL
OpenGL.FORWARD_COMPATIBLE_ONLY = True
OpenGL.ERROR_ON_COPY = True
from OpenGL import GL as GL
from PySide import QtCore, QtGui, QtOpenGL

from freud import qtmanager;
from freud.viz.render import gl

null = c_void_p(0)

class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, scene, *args, **kwargs):
        QtOpenGL.QGLWidget.__init__(self, *args, **kwargs)
        self.scene = scene;

    def resizeGL(self, w, h):
        GL.glViewport(0, 0, w, h)
        
    def paintGL(self):
        GL.glClearColor(0.0, 0.0, 0.0, 0.0);
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self.draw_gl.draw(self.scene);

    def initializeGL(self):
        print("OpenGL version: ", GL.glGetString(GL.GL_VERSION))
        self.draw_gl = gl.DrawGL();
        
class Window(QtGui.QWidget):
    def __init__(self, scene, *args, **kwargs):
        QtGui.QWidget.__init__(self, *args, **kwargs)

        self.glWidget = GLWidget(scene)

        mainLayout = QtGui.QHBoxLayout()
        mainLayout.addWidget(self.glWidget)
        self.setLayout(mainLayout)

        self.setWindowTitle("Hello World")


##########################################
## Module init
# initialize Qt and set the default GL format
qtmanager.initApp();
glFormat = QtOpenGL.QGLFormat();
glFormat.setVersion(3, 2);
glFormat.setProfile( QtOpenGL.QGLFormat.CompatibilityProfile );
glFormat.setSampleBuffers(True);
QtOpenGL.QGLFormat.setDefaultFormat(glFormat);
