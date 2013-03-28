from __future__ import division, print_function
import numpy
import math
from ctypes import c_void_p
import OpenGL
#OpenGL.FULL_LOGGING = True
OpenGL.FORWARD_COMPATIBLE_ONLY = True
OpenGL.ERROR_ON_COPY = True
from OpenGL import GL

null = c_void_p(0)

## \package freud.viz.render.gl
#
# GL output for freud.viz
#

## \internal
# dict of strings holding all the vertex shaders
_vertex_shaders = {};

# Disks vertex shader
_vertex_shaders['Disks'] = """
#version 120

attribute vec4 position;
attribute vec2 mapcoord;
attribute vec4 color;

varying vec4 v_mapcoord;
varying vec4 v_color;
void main()
    {
    gl_Position = position;
    v_color = color;
    v_mapcoord = mapcoord;
    }
""";

## \internal
# dict of strings holding all the fragment shaders
_fragment_shaders = {};

_fragment_shaders['Disks'] = """
#version 120

varying vec4 v_mapcoord;
varying vec4 v_color;
void main()
    {
    vec4 gamma = vec4(1.0 / 1.0);
    gamma.w = 1.0;
    gl_FragColor = pow(vColor, gamma);
    }
""";

## \internal
# dict of lists listing the attributes (in order) for each type of shader program
# Order is important, because it defines the order of indices passed to glVertexAttribPointer
_attributes = {};

_attributes['Disks'] = ['position', 'mapcoord', 'color'];

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
# identical. This is why primitives encourage recreation of primitives and not changing the data. However, this is
# only a temporary solution. For example, recreating the entire primitive is wasteful when just changing the outline
# width.
# 
# TODO - add some kind of dirty flag to primitives and set commands necessary to update values that set the flag.
#
class DrawGL(object):
    ## Initialize a DrawGL
    #
    def __init__(self):
        # initialize programs
        self.programs = {};
        for t in _vertex_shaders.keys():
            self.programs[t] = Program(_vertex_shaders[t], _fragment_shaders[t], _attributes[t]);
        
        self.cache = Cache();

    ## Draws a primitive to the GL context
    # \param prim Primitive to draw
    #
    def draw_Primitive(self, prim):
        raise RuntimeError('DrawGL encountered an unknown primitive type');

    ## Draw an entire scene
    # \param scene Scene to write
    #
    def draw_Scene(self, scene):
        # setup the camera matrix
        
        
        # loop through the render primitives and write out each one
        for i,group in enumerate(scene.groups):
            # apply the group transformation matrix
            for j,primitive in enumerate(group.primitives):
                self.write(primitive)

    ## Draw disks
    # \param disks Disks to write
    #
    def draw_Disks(self, disks):
        # get the geometry from the cache
        data = self.cache.get(disks, CacheDisks)
        
        # bind everything and then draw the disks
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, data.buffer_position);
        GL.glEnableVertexAttribArray(0);
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, c_void_p(0));
        
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, data.buffer_mapcoord);
        GL.glEnableVertexAttribArray(1);
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, c_void_p(0));

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, data.buffer_color);
        GL.glEnableVertexAttribArray(2);
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, c_void_p(0));
        
        GL.glUseProgram(self.program);
        
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, data.N*6);

        # unbind everything we bound
        GL.glDisableVertexAttribArray(0);
        GL.glDisableVertexAttribArray(1);
        GL.glDisableVertexAttribArray(2);
        GL.glUseProgram(0);

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
    def write(self, obj):
        meth = None;
        for cls in obj.__class__.__mro__:
            meth_name = 'draw_'+cls.__name__;
            meth = getattr(self, meth_name, None);
            if meth is not None:
                break;

        if meth is None:
            raise RuntimeError('DrawGL does not know how to write a {0}'.format(obj.__class__.__name__));
        return meth(obj);

## Cache manager
#
# The cache manager takes care of initializing the CacheItem instances as needed and saving them for later use.
#
# TODO: need some kind of frame start/stop mechanism so that the manager can tell when to free unused resources
#
class Cache(object):
    ## Initialize the cache manager
    # Initially the manager is empty
    def __init__(self):
        self.cache = {};
    
    ## Load a value out of the cache
    # \param prim Primitive to load
    # \param typ Class type of the CacheItem
    #
    # If \a prim is already in the cache, return its CacheItem immediately. Otherwise, instantiate it and then return
    # it. Items are cached based on their ident value, which is unique among generated primitives. Once a primitive is
    # cached, it is never updated.
    #
    def get(self, prim, typ):
        if prim.ident not in self.cache:
            self.cache[prim.ident] = typ(prim);
        
        return self.cache[prim.ident];

## Base class for cache items
#
# DrawGL stores openGL geometry in a cache. Each item stored in the cache derives from CacheItem and implements the same
# interface. There are a few requirements. 1) CacheItems are only created or access while the OpenGL context is active.
# 2) They initialize their geometry (or other OpenGL entities) on initialization. 3) The __del__() method
# releases all of the OpenGL entities.
#
# Other than that, CacheItems are free-form and can be implemented however needed for the specific primitive. Typical
# use-cases will probably just store a few buffers as member variables to be directly accessed by DrawGL calls.
#
# Putting the code for geometry generation here keeps it compartmentalized separately from the code specific to drawing
# the actual geometry. Of course, drawing and the geometry format are tied closely and both bits of code will need to be
# updated in tandem.
#
class CacheItem(object):
    ## Initialize a cache item
    # \param prim Primitive to represent
    #
    def __init__(self, prim):
        pass
    
    ## Release a cache item
    # 
    # Frees all OpenGL resources used by the item
    #
    def __del__(self):
        pass

## Cache RepeatedPolygon geometry
#
# Store the OpenGL geometry for the RepeatedPolygon primitive
#
class CacheRepeatedPolygons(CacheItem):
    ## Initialize a cache item
    # \param prim Primitive to represent
    #
    def __init__(self, prim):
        CacheItem.__init__(self);
    
    ## Release a cache item
    # 
    # Frees all OpenGL resources used by the item
    #
    def __del__(self):
        pass  

## Cache Disk geometry
#
# Store the OpenGL geometry for the Disk primitive.
#
# Disks are rendered using 2 triangles covering a square area with a width equal to the diameter of the disk.
# Coordinates in the range from -d/2 to d/2 are passed in another buffer and interpolated across the triangles for the
# fragment shader to use in drawing the outline (where d is the diameter of the disk). The outline width is a global
# parameter. This is drawn with glDrawArrays(GL_TRIANGLES).
#
# All variables are directly accessible class members.
#  - N: number of disks
#  - outline: outline width (in map coordinates)
#  - buffer_position: OpenGL buffer (N*6 2-element positions)
#  - buffer_mapcoord: OpenGL buffer (N*6 2-element map coordinates)
#  - buffer_color: OpenGL buffer (N*6 4-element colors)
#
# There are certainly improvements one could make, but this seems like the simplest place to start and the least
# error prone. Ideas for improvement:
#  - one triangle per disk (3 verts)
#  - 4 verts per disk with element based drawing
#  - point sprites (not sure how to get these working, already tried some)
#  - geometry shaders (requires opengl > 2.1, which we don't have on Mac right now)
#
class CacheDisks(CacheItem):
    ## Initialize a cache item
    # \param prim Primitive to represent
    #
    def __init__(self, prim):
        CacheItem.__init__(self);
        
        # simple scalar values
        self.N = len(prim.positions);
        self.outline = prim.outline;
        
        # initialize values for buffers
        position = numpy.zeros(shape=(self.N, 6, 2), dtype=numpy.float32);
        mapcoord = numpy.zeros(shape=(self.N, 6, 2), dtype=numpy.float32);
        color = numpy.zeros(shape=(self.N, 6, 4), dtype=numpy.float32);
        
        # start all coords at the center, with all the same color
        for i in range(6):
            position[:,i,:] = prim.positions;
            color[:,i,:] = prim.color;
        
        # Map of coords
        # 0 --- 2  5
        # |    /  /|
        # |   /  / |
        # |  /  /  |
        # | /  /   |
        # |/  /    |
        # 1  3 --- 4
        #
        
        # Update x coordinates
        for i in [0,1,3]:
            position[:,i,0] -= prim.diameters/2;
            mapcoord[:,i,0] = -prim.diameters/2;
        
        for i in [2,4,5]:
            position[:,i,0] += prim.diameters/2;
            mapcoord[:,i,0] = prim.diameters/2;

        # update y coordinates
        for i in [0,2,5]:
            position[:,i,1] += prim.diameters/2;
            mapcoord[:,i,1] = prim.diameters/2;

        for i in [1,3,4]:
            position[:,i,1] -= prim.diameters/2;
            mapcoord[:,i,1] = -prim.diameters/2;
        
        # generate OpenGL buffers and copy data
        self.buffer_position = GL.glGenBuffers(1);
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffer_position);
        GL.glBufferData(GL.GL_ARRAY_BUFFER, position, GL.GL_STATIC_DRAW);

        self.buffer_mapcoord = GL.glGenBuffers(1);
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffer_mapcoord);
        GL.glBufferData(GL.GL_ARRAY_BUFFER, mapcoord, GL.GL_STATIC_DRAW);

        self.buffer_color = GL.glGenBuffers(1);
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffer_color);
        GL.glBufferData(GL.GL_ARRAY_BUFFER, color, GL.GL_STATIC_DRAW);

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0);
    
    ## Release a cache item
    # 
    # Frees all OpenGL resources used by the item
    #
    def __del__(self):
        print('Deleting disk cache: ', id(self));
        buf_list = numpy.array([self.buffer_position, self.buffer_mapcoord, self.buffer_color], dtype=numpy.uint32);
        GL.glDeleteBuffers(3, buf_list);

## OpenGL program
#
# Lightweight class interface for initializing, querying, and setting parameters on OpenGL programs
#
class Program(object):
    ## Initialize the program
    # \param vertex_shader
    # \param fragment_shader
    # \param attributes
    #
    # Compiles and initializes the Program. After initialization, the program attribute is accessible and usable
    # as an OpenGL program
    #
    def __init__(self, vertex_shader, fragment_shader, attributes):
        self.program = self._initialize_program(vertex_shader, fragment_shader, attributes);
    
    ## Clean up
    # Release OpenGL resources
    #
    def __del__(self):
        print('Deleting program: ', self.program);
        GL.glDeleteProgram(self.program);
    
    @staticmethod
    def _initialize_program(vertex_shader, fragment_shader, attributes):
        shaders = [];
        
        shaders.append(self._create_shader(GL.GL_VERTEX_SHADER, vertex_shader));
        shaders.append(self._create_shader(GL.GL_FRAGMENT_SHADER, fragment_shader));
        
        program = self._create_program(shaders, attributes);

        for shader in shaders:
            GL.glDeleteShader(shader);
        
        return program;

    @staticmethod
    def _create_shader(stype, source):
        shader = GL.glCreateShader(type);
        GL.glShaderSource(shader, source);
        
        GL.glCompileShader(shader);
        
        status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS, None);
        if status == GL.GL_FALSE:
            msg = GL.glGetShaderInfoLog(shader);
            err = "Error compiling shader: " + msg;
            raise RuntimeError(err);

        return shader;

    @staticmethod
    def _create_program(shaders, attributes):
        program = GL.glCreateProgram();
        
        for i,attrib in enumerate(attributes):
            GL.glBindAttribLocation(program, i, attrib);
        
        for shader in shaders:
            GL.glAttachShader(program, shader);
        
        GL.glLinkProgram(program);
        
        status = GL.glGetProgramiv(program, GL.GL_LINK_STATUS, None);
        if status == GL.GL_FALSE:
            msg = GL.glGetProgramInfoLog(shader);
            err = "Error compiling shader: " + msg;
            raise RuntimeError(err);
        
        for shader in shaders:
            GL.glDetachShader(program, shader);
        
        return program;
