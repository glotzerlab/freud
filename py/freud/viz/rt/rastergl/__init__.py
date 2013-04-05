from __future__ import division, print_function

import numpy
import math
from ctypes import c_void_p
from OpenGL import GL as gl

from . import glprimitive

null = c_void_p(0)

## \package freud.viz.rt.rastergl
#
# Real time GL rasterization for freud.viz
#

## DrawGL draws scenes using OpenGL
#
# Instantiating a DrawGL loads shaders and performs other common init tasks. You can then call draw() as many times as
# you want to draw GL frames.
#
# DrawGL uses the visitor pattern to handle output methods for different primitives. 
# The method used is described here: http://peter-hoffmann.com/2010/extrinsic-visitor-pattern-python-inheritance.html
#
# Internally, GL geometry differs from the raw primitive data. DrawGL generates this geometry on the fly as needed
# and stores it in a cache. The next draw call will reuse geometry data out of the cache for primitives that are
# identical. This is why primitives encourage recreation of primitives and not changing the data. 
# 
# TODO - add some kind of dirty flag to primitives and set commands necessary to update values that set the flag.
#
class DrawGL(object):
    ## Initialize a DrawGL
    #
    def __init__(self):
        # initialize programs
        class_list = [glprimitive.GLDisks];
        self.programs = {};
        for cls in class_list:
            self.programs[cls] = Program(cls);
        
        self.cache = glprimitive.Cache();

    ## Draws a primitive to the GL context
    # \param prim Primitive to draw
    #
    def draw_Primitive(self, prim):
        raise RuntimeError('DrawGL encountered an unknown primitive type');

    ## Draw an entire scene
    # \param scene Scene to write
    #
    def draw_Scene(self, scene):
        # setup the camera
        self.camera = scene.camera;
        
        # loop through the render primitives and write out each one
        for i,group in enumerate(scene.groups):
            # apply the group transformation matrix
            for j,primitive in enumerate(group.primitives):
                self.draw(primitive)

    ## Draw disks
    # \param prim Disks to write
    #
    def draw_Disks(self, prim):
        # shorthand for the class type of the GLPrimitive
        cls = glprimitive.GLDisks;
        program = self.programs[cls].program;
        
        # get the geometry from the cache and draw it
        glprim = self.cache.get(prim, cls);
        glprim.draw(program, self.camera);

    ## Draw repeated polygons
    # \param polygons Polygons to draw
    #
    def draw_RepeatedPolygons(self, polygons):
        pass

    ## Write out image
    # \param img Image to write
    #
    def draw_Image(self, img):
        pass
        
    ## Draw a viz element
    # \param obj Object to write
    #
    def draw(self, obj):
        meth = None;
        for cls in obj.__class__.__mro__:
            meth_name = 'draw_'+cls.__name__;
            meth = getattr(self, meth_name, None);
            if meth is not None:
                break;

        if meth is None:
            raise RuntimeError('DrawGL does not know how to write a {0}'.format(obj.__class__.__name__));
        return meth(obj);

## OpenGL program
#
# Lightweight class interface for initializing, querying, and setting parameters on OpenGL programs
#
class Program(object):
    ## Initialize the program from a GLPrimitive
    # \param glprim GLPrimitive to compile
    # Compiles and initializes the Program. After initialization, the program attribute is accessible and usable
    # as an OpenGL program
    #
    def __init__(self, glprim=None):
        self.program = self._initialize_program(glprim.vertex_shader, glprim.fragment_shader, glprim.attributes);
    
    ## Clean up
    # Release OpenGL resources
    #
    def __del__(self):
        gl.glDeleteProgram(self.program);
    
    @staticmethod
    def _initialize_program(vertex_shader, fragment_shader, attributes):
        shaders = [];
        
        shaders.append(Program._create_shader(gl.GL_VERTEX_SHADER, vertex_shader));
        shaders.append(Program._create_shader(gl.GL_FRAGMENT_SHADER, fragment_shader));
        
        program = Program._create_program(shaders, attributes);

        for shader in shaders:
            gl.glDeleteShader(shader);
        
        return program;

    @staticmethod
    def _create_shader(stype, source):
        shader = gl.glCreateShader(stype);
        gl.glShaderSource(shader, source);
        
        gl.glCompileShader(shader);
        
        status = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS, None);
        if status == gl.GL_FALSE:
            msg = gl.glGetShaderInfoLog(shader);
            err = "Error compiling shader: " + msg;
            raise RuntimeError(err);

        return shader;

    @staticmethod
    def _create_program(shaders, attributes):
        program = gl.glCreateProgram();
        
        for i,attrib in enumerate(attributes):
            gl.glBindAttribLocation(program, i, attrib);
        
        for shader in shaders:
            gl.glAttachShader(program, shader);
        
        gl.glLinkProgram(program);
        
        status = gl.glGetProgramiv(program, gl.GL_LINK_STATUS, None);
        if status == gl.GL_FALSE:
            msg = gl.glGetProgramInfoLog(shader);
            err = "Error compiling shader: " + msg;
            raise RuntimeError(err);
        
        for shader in shaders:
            gl.glDetachShader(program, shader);
        
        return program;
