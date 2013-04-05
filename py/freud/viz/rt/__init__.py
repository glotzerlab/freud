from __future__ import division, print_function

import numpy
import math
import time
from ctypes import c_void_p
import logging
logger = logging.getLogger(__name__);

# pyside and OpenGL are required deps for this module (not for all of freud), but we don't want to burden user scripts
# with lots of additional imports. So try and import them and throw a warning up to the parent module to handle
try:
    import OpenGL
    from PySide import QtCore, QtGui, QtOpenGL
except ImportError:
    logger.warning('Either PySide or pyopengl is not available, aborting rt initialization');
    raise ImportWarning('PySide or pyopengl not available');

# set opengl options
OpenGL.FORWARD_COMPATIBLE_ONLY = True
OpenGL.ERROR_ON_COPY = True
OpenGL.INFO_LOGGING = False

# force gl logger to emit only warnings and above
gl_logger = logging.getLogger('OpenGL')
gl_logger.setLevel(logging.WARNING)

from OpenGL import GL as gl

from freud import qt;
from . import rastergl

## \package freud.viz.rt
#
# Real-time visualization Qt widgets and rendering routines. 
#
# \note freud.viz.rt **requires** pyside and pyopengl. If these dependencies are not present, a warning is issued to the
# logger, but execution continues with freud.viz.rt = None.
#

null = c_void_p(0)

class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, scene, *args, **kwargs):
        QtOpenGL.QGLWidget.__init__(self, *args, **kwargs)
        self.scene = scene;
        
        # timers for FPS
        self.last_time = time.time();
        self.frame_count = 0;
        self.timer_fps = QtCore.QTimer(self)
        self.timer_fps.timeout.connect(self.updateFPS)
        self.timer_fps.start(500)
        
    def resizeGL(self, w, h):
        gl.glViewport(0, 0, w, h)
        self.scene.camera.setAspect(w/h);
        self.scene.camera.resolution = h;
        
    def paintGL(self):
        self.frame_count += 1;
        
        gl.glClearColor(1.0, 1.0, 1.0, 0.0);
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT);
        self.draw_gl.draw(self.scene);

    def initializeGL(self):
        logger.info('OpenGL version: ' + gl.glGetString(gl.GL_VERSION))
        self.draw_gl = rastergl.DrawGL();
    
    def updateFPS(self):
        cur_time = time.time();

        if self.frame_count > 0:
            elapsed_time = cur_time - self.last_time;
            print(self.frame_count / elapsed_time, "FPS");
            self.frame_count = 0;
            
        self.last_time = cur_time;
    
        
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

# initialize Qt application
qt.init_app();

# set the default GL format
glFormat = QtOpenGL.QGLFormat();
glFormat.setVersion(2, 1);
glFormat.setProfile( QtOpenGL.QGLFormat.CompatibilityProfile );
glFormat.setSampleBuffers(True);
glFormat.setSwapInterval(0);
QtOpenGL.QGLFormat.setDefaultFormat(glFormat);
