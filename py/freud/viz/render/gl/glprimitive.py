from __future__ import division, print_function
import numpy
import math
from ctypes import c_void_p
from OpenGL import GL as gl

null = c_void_p(0)

## Cache manager
#
# The cache manager takes care of initializing the GLPrimitive instances as needed and saving them for later use.
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

## Base class for cached GLPrimitive items
#
# DrawGL stores openGL geometry in a cache. Each item stored in the cache derives from GLPrimitive and implements the same
# interface. There are a few requirements. 1) GLPrimitives are only created or access while the OpenGL context is active.
# 2) They initialize their geometry (or other OpenGL entities) on initialization. 3) The __del__() method
# releases all of the OpenGL entities. 4) They provide a class static vertex_shader, fragment_shader, and attributes.
# 5) They have a draw() method which draws the primitive. 
#
# Other than that, GLPrimitives are free-form and can be implemented however needed for the specific primitive. Typical
# use-cases will probably just store a few buffers as member variables to be directly accessed by DrawGL calls.
#
# Putting the code for shaders, geometry generation, and drawing here keeps it compartmentalized separately from 
# everything else. Since DrawGL has one method per primitive type, the arguments and interface to draw() can 
# even differ from primitive to primitive - though there should be some generic standard that most follow for
# consistency. That generic model will evolve as we add cameras, lights, and materials.
#
# Shaders are stored as class static variables. DrawGL compiles and stores all shader programs on initialization.
# The program index is passed to the draw() method for use in querying parameters, setting the program, etc...
# 
class GLPrimitive(object):
    ## The vertex shader for this primitive type
    vertex_shader = '';
    
    ## The fragment shader for this primitive type
    fragment_shader = '';
    
    ## List of attributes for this primitive type's shaders
    # list the attributes (in order) for each type of shader program
    # The order is important, because it defines the order of indices passed to glVertexAttribPointer
    attributes = [];
    
    ## Initialize a cached GL primitive
    # \param prim base Primitive to represent
    #
    def __init__(self, prim):
        pass
    
    ## Draw the primitive
    #
    def draw(self):
        pass
    
    ## Release a cached GL primitive
    # 
    # Frees all OpenGL resources used by the primitive
    #
    def __del__(self):
        pass

## RepeatedPolygon geometry
#
# Store and draws the OpenGL geometry for the RepeatedPolygon primitive
#
class GLRepeatedPolygons(GLPrimitive):
    ## Initialize a cached GL primitive
    # \param prim base Primitive to represent
    #
    def __init__(self, prim):
        CacheItem.__init__(self);
    
    ## Release a cached GL primitive
    # 
    # Frees all OpenGL resources used by the primitive
    #
    def __del__(self):
        pass  

## Disk geometry
#
# Store and draw the OpenGL geometry for the Disk primitive.
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
#  - geometry shaders (requires opengl > 2.1, which we don't have on Mac until Qt5 and supporting python interfaces are
#                      easily available (macports))
#
class GLDisks(GLPrimitive):
    ## Vertex shader for drawing disks
    #
    # Transform the incoming verts by the camera and pass through everything else.
    vertex_shader = """
#version 120

uniform mat4 camera;

attribute vec4 position;
attribute vec2 mapcoord;
attribute vec4 color;

varying vec4 v_mapcoord;
varying vec4 v_color;
void main()
    {
    gl_Position = camera * position;
    v_color = color;
    v_mapcoord = mapcoord;
    }
""";

    ## Fragment shader for drawing disks
    #
    # Disks are rendered in 2D mode, so no gamma correction is needed (colors are just passed through directly).
    fragment_shader = """
#version 120

varying vec4 v_mapcoord;
varying vec4 v_color;
void main()
    {
    gl_FragColor = v_color;
    }
""";
    
    ## Attributes for drawing disks
    attributes = ['position', 'mapcoord', 'color'];
    
    ## Initialize a cached GL primitive
    # \param prim base Primitive to represent
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
        self.buffer_position = gl.glGenBuffers(1);
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_position);
        gl.glBufferData(gl.GL_ARRAY_BUFFER, position, gl.GL_STATIC_DRAW);

        self.buffer_mapcoord = gl.glGenBuffers(1);
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_mapcoord);
        gl.glBufferData(gl.GL_ARRAY_BUFFER, mapcoord, gl.GL_STATIC_DRAW);

        self.buffer_color = gl.glGenBuffers(1);
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_color);
        gl.glBufferData(gl.GL_ARRAY_BUFFER, color, gl.GL_STATIC_DRAW);

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0);
    
    ## Draw the primitive
    # \param program OpenGL shader program
    # \param camera The camera to use when drawing
    #
    def draw(self, program, camera):
        gl.glUseProgram(program);
        
        # update the camera matrix
        camera_uniform = gl.glGetUnifromLocation(program, "camera");
        gl.glUniformMatrix4fv(camera_uniform, 1, True, camera.ortho_2d_matrix);
                
        # bind everything and then draw the disks
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_position);
        gl.glEnableVertexAttribArray(0);
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, c_void_p(0));
        
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_mapcoord);
        gl.glEnableVertexAttribArray(1);
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, c_void_p(0));

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_color);
        gl.glEnableVertexAttribArray(2);
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, c_void_p(0));
        
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.N*6);

        # unbind everything we bound
        gl.glDisableVertexAttribArray(0);
        gl.glDisableVertexAttribArray(1);
        gl.glDisableVertexAttribArray(2);
        gl.glUseProgram(0);
    
    ## Release a cached GL primitive
    # 
    # Frees all OpenGL resources used by the primitive
    #
    def __del__(self):
        print('Deleting disk cache: ', id(self));
        buf_list = numpy.array([self.buffer_position, self.buffer_mapcoord, self.buffer_color], dtype=numpy.uint32);
        gl.glDeleteBuffers(3, buf_list);
