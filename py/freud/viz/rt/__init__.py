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
# OpenGL.ERROR_CHECKING = False

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
    # animation states the UI code can take
    ANIM_IDLE = 1;
    ANIM_PAN = 2;
    
    
    def __init__(self, scene, *args, **kwargs):
        QtOpenGL.QGLWidget.__init__(self, *args, **kwargs)
        self.scene = scene;
        
        self.setCursor(QtCore.Qt.OpenHandCursor)
        
        # initialize state machine variables
        self._anim_state = GLWidget.ANIM_IDLE;
        self._prev_pos = numpy.array([0,0], dtype=numpy.float32);
        self._prev_time = time.time();
        self._pan_vel = numpy.array([0,0], dtype=numpy.float32);
        self._initial_pan_vel = numpy.array([0,0], dtype=numpy.float32);
        self._initial_pan_time = time.time();
        
        # timers for FPS
        self.last_time = time.time();
        self.frame_count = 0;
        self.timer_fps = QtCore.QTimer(self)
        self.timer_fps.timeout.connect(self.updateFPS)
        #self.timer_fps.start(500)
        
        self._timer_animate = QtCore.QTimer(self)
        self._timer_animate.timeout.connect(self.animate)
        
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

    def animate(self):
        if self._anim_state == GLWidget.ANIM_IDLE:
            self._timer_animate.stop();
        
        if self._anim_state == GLWidget.ANIM_PAN:
            cur_time = time.time();
            self._pan_vel = numpy.exp(-(cur_time - self._initial_pan_time)/0.1) * self._initial_pan_vel;
            
            delta = self._pan_vel * (cur_time - self._prev_time) * self.scene.camera.pixel_size;
            delta[0] = delta[0] * -1;
            self.scene.camera.position[0:2] += delta;
            
            self._prev_time = cur_time;
                                    
            if numpy.dot(self._pan_vel, self._pan_vel) < 100:
                self._anim_state = GLWidget.ANIM_IDLE;
            
            self.updateGL();
    
    def updateFPS(self):
        cur_time = time.time();

        if self.frame_count > 0:
            elapsed_time = cur_time - self.last_time;
            print(self.frame_count / elapsed_time, "FPS");
            self.frame_count = 0;
            
        self.last_time = cur_time;
    
    def closeEvent(self, event):
        # stop the animation loop
        self._anim_state = GLWidget.ANIM_IDLE;
    
    def keyPressEvent(self, event):
        pass
        
    def keyReleaseEvent(self, event):
        pass
        
    def mouseMoveEvent(self, event):
        # update the camera position based on the movement from the previous position
        # while dragging with the left mouse button
        if event.buttons() & QtCore.Qt.LeftButton:
            # update camera position
            cur_time = time.time();
            cur_pos = numpy.array([event.x(), event.y()], dtype=numpy.float32);
            delta = (cur_pos - self._prev_pos) * self.scene.camera.pixel_size;
            delta[0] = delta[0] * -1;
            self.scene.camera.position[0:2] += delta;
            
            # compute pan velocity in camera pixels/second
            self._pan_vel[:] = (cur_pos - self._prev_pos) / (cur_time - self._prev_time);
            
            self._prev_time = cur_time;
            self._prev_pos = cur_pos;
            self.updateGL();
            event.accept();
        else:
            event.ignore();
    
    def mousePressEvent(self, event):
        # start mouse-control panning the camera when the left button is pressed
        if event.button() == QtCore.Qt.LeftButton:
            self._anim_state = GLWidget.ANIM_IDLE;
            self._prev_pos[0] = event.x();
            self._prev_pos[1] = event.y();
            self._prev_time = time.time();
            self._pan_vel[:] = 0;
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept();
        else:
            event.ignore();
    
    def mouseReleaseEvent(self, event):
        # stop mouse-control panning the camera when the left button is released
        # and start the animated panning
        if event.button() == QtCore.Qt.LeftButton:
            self._anim_state = GLWidget.ANIM_PAN;
            self._initial_pan_vel[:] = self._pan_vel[:];
            self._initial_pan_time = self._prev_time;
            self._timer_animate.start()
            self.setCursor(QtCore.Qt.OpenHandCursor)
            event.accept();
        else:
            event.ignore();
    
    def wheelEvent(self, event):
        # control speed based on modifiers (ctrl = slow)
        if event.modifiers() == QtCore.Qt.ControlModifier:
            speed = 0.05;
        else:
            speed = 0.2;
        
        if event.orientation() == QtCore.Qt.Vertical:
            # zoom based on the mouse wheel
            f = 1 - speed * float(event.delta())/120;
            
            self.scene.camera.setHeight(self.scene.camera.getHeight() * f);
            self.updateGL();
            event.accept();
    
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
