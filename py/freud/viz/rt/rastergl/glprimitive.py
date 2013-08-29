from __future__ import division, print_function
import numpy
import math
import time
from ctypes import c_void_p
from OpenGL import GL as gl
from PySide import QtGui

null = c_void_p(0)

## \internal
# \package freud.viz.rt.rastergl.glprimitive
#
# Real time GL rasterization for freud.viz. Classes in glprimitive are for internal use by DrawGL only and their
# interface may change at any time.
#

## Cache manager
# \note Cache is used internally by DrawGL and is not part of the public freud interface
#
# The cache manager takes care of initializing the GLPrimitive instances as needed and saving them for later use.
#
# startFrame() signifies the start of a new frame. stopFrame signifies the end of a frame. The cache uses information
# about entities accessed during a frame to decide which ones to evict from the cache.
#
class Cache(object):
    ## Initialize the cache manager
    # Initially the manager is empty
    def __init__(self):
        self.cache = {};
        self.accessed_ids = set();

    ## Start a frame
    # Notify the cache that a frame render is starting
    def startFrame(self):
        self.accessed_ids.clear()

    ## End a frame
    # Notify the cache that a frame render has completed. The cache may choose to free OpenGL resources at this time.
    # An OpenGL context must be active when calling endFrame();
    def endFrame(self):
        to_delete = [];

        for ident in self.cache.keys():
            if not ident in self.accessed_ids:
                self.cache[ident].destroy();
                to_delete.append(ident);

        for ident in to_delete:
            del self.cache[ident]

    ## Destroy OpenGL resources
    # OpenGL calls need to be made when a context is active. This class provides an explicit destroy() method so that
    # resources can be released at a controlled time. (not whenever python decides to call __del__).
    #
    def destroy(self):
        for c in self.cache.values():
            c.destroy();

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
        elif prim.updated:
            self.cache[prim.ident].update(prim, prim.updated)
            prim.updated = []

        self.accessed_ids.add(prim.ident);

        return self.cache[prim.ident];


## Base class for cached primitives
# \note GLPrimitive is used internally by DrawGL and is not part of the public freud interface
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

    ## Destroy OpenGL resources
    # OpenGL calls need to be made when a context is active. This class provides an explicit destroy() method so that
    # resources can be released at a controlled time. (not whenever python decides to call __del__.
    #
    def destroy(self):
        pass

## Disk geometry
# \note GLDisks is used internally by DrawGL and is not part of the public freud interface
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
#  - buffer_mapcoord: OpenGL buffer (N*6 3-element map coordinates)
#  - buffer_color: OpenGL buffer (N*6 4-element colors)
#
# mapcoord.xy stores the position in the disk and z stores the diameter of the disk (need to get it in there somehow).
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
uniform float outline_width;

attribute vec4 position;
attribute vec3 mapcoord;
attribute vec4 color;

varying vec3 v_mapcoord;
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

uniform float outline;
uniform float pixel_size;

varying vec3 v_mapcoord;
varying vec4 v_color;
void main()
    {
    // determine position in the disk (in pixels) and its radius
    vec2 p = v_mapcoord.xy / pixel_size;
    float r = sqrt(dot(p,p));
    float disk_r = v_mapcoord.z/2.0f / pixel_size;

    // determine the edges of the various colors
    float color_r = disk_r - outline / pixel_size;

    // color the output fragment appropriately
    vec4 color_inside = v_color;
    vec4 color_edge = vec4(0,0,0,v_color.a);
    if (r < color_r-1)
        {
        gl_FragColor = color_inside;
        }
    else if (r < color_r+1)
        {
        // antialias color-edge boundary
        float d = color_r + 1 - r;
        float a = exp2(-2 * d * d);
        gl_FragColor = a * color_edge + (1-a) * color_inside;
        }
    else if (r < disk_r - 1)
        gl_FragColor = color_edge;
    else if (r < disk_r + 1)
        {
        // antialias color-edge boundary (alpha blending is used for the blend)
        float d = disk_r - 1 - r;
        color_edge.a *= exp2(-2 * d * d);
        gl_FragColor = color_edge;
        }
    else
        discard;
    }
""";

    ## Attributes for drawing disks
    attributes = ['position', 'mapcoord', 'color'];

    ## Initialize a cached GL primitive
    # \param prim base Primitive to represent
    #
    def __init__(self, prim):
        GLPrimitive.__init__(self, prim);

        # simple scalar values
        self.N = len(prim.positions);
        self.outline = prim.outline;

        # initialize values for buffers
        position = numpy.zeros(shape=(self.N, 6, 2), dtype=numpy.float32);
        mapcoord = numpy.zeros(shape=(self.N, 6, 3), dtype=numpy.float32);
        color = numpy.zeros(shape=(self.N, 6, 4), dtype=numpy.float32);

        self.build_geometry(position, mapcoord, color, prim.positions, prim.colors, prim.diameters);

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

    ## Fast kernel to generate disk geometry
    #
    @staticmethod
    # @numba.jit('f4(f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:,:], f4[:,:], f4[:])', nopython=True)
    def build_geometry(position, mapcoord, color, positions_in, colors_in, diameters_in):
        N = position.shape[0]

        # start all coords at the center, with all the same color
        for i in range(N):
            for j in range(6):
                position[i,j,0] = positions_in[i,0];
                position[i,j,1] = positions_in[i,1];

                color[i,j,0] = colors_in[i,0];
                color[i,j,1] = colors_in[i,1];
                color[i,j,2] = colors_in[i,2];
                color[i,j,3] = colors_in[i,3];

        # Map of coords
        # 0 --- 2  5
        # |    /  /|
        # |   /  / |
        # |  /  /  |
        # | /  /   |
        # |/  /    |
        # 1  3 --- 4
        #

        # Expand the size of the quad by 5% to leave room for anti-alias pixel rendering on the edge
        # this doesn't guarantee that enough will be rendered, but should work in most cases
        ex_factor = 1.05;

        # Update x coordinates
        for i in range(N):
            position[i,0,0] -= diameters_in[i]/2 * ex_factor;
            mapcoord[i,0,0] = -diameters_in[i]/2 * ex_factor;
            mapcoord[i,0,2] = diameters_in[i];

            position[i,1,0] -= diameters_in[i]/2 * ex_factor;
            mapcoord[i,1,0] = -diameters_in[i]/2 * ex_factor;
            mapcoord[i,1,2] = diameters_in[i];

            position[i,3,0] -= diameters_in[i]/2 * ex_factor;
            mapcoord[i,3,0] = -diameters_in[i]/2 * ex_factor;
            mapcoord[i,3,2] = diameters_in[i];

            position[i,2,0] += diameters_in[i]/2 * ex_factor;
            mapcoord[i,2,0] = diameters_in[i]/2 * ex_factor;
            mapcoord[i,2,2] = diameters_in[i];

            position[i,4,0] += diameters_in[i]/2 * ex_factor;
            mapcoord[i,4,0] = diameters_in[i]/2 * ex_factor;
            mapcoord[i,4,2] = diameters_in[i];

            position[i,5,0] += diameters_in[i]/2 * ex_factor;
            mapcoord[i,5,0] = diameters_in[i]/2 * ex_factor;
            mapcoord[i,5,2] = diameters_in[i];

            # # update y coordinates
            position[i,0,1] += diameters_in[i]/2 * ex_factor;
            mapcoord[i,0,1] = diameters_in[i]/2 * ex_factor;

            position[i,2,1] += diameters_in[i]/2 * ex_factor;
            mapcoord[i,2,1] = diameters_in[i]/2 * ex_factor;

            position[i,5,1] += diameters_in[i]/2 * ex_factor;
            mapcoord[i,5,1] = diameters_in[i]/2 * ex_factor;

            position[i,1,1] -= diameters_in[i]/2 * ex_factor;
            mapcoord[i,1,1] = -diameters_in[i]/2 * ex_factor;

            position[i,3,1] -= diameters_in[i]/2 * ex_factor;
            mapcoord[i,3,1] = -diameters_in[i]/2 * ex_factor;

            position[i,4,1] -= diameters_in[i]/2 * ex_factor;
            mapcoord[i,4,1] = -diameters_in[i]/2 * ex_factor;

        return 0;

    ## Draw the primitive
    # \param program OpenGL shader program
    # \param camera The camera to use when drawing
    #
    def draw(self, program, camera):
        # save state
        gl.glPushAttrib(gl.GL_ENABLE_BIT | gl.GL_COLOR_BUFFER_BIT);

        # setup state
        gl.glDisable(gl.GL_MULTISAMPLE);
        gl.glEnable(gl.GL_BLEND);
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA);

        gl.glUseProgram(program);

        # update the camera matrix
        camera_uniform = gl.glGetUniformLocation(program, b'camera');
        gl.glUniformMatrix4fv(camera_uniform, 1, True, camera.ortho_2d_matrix);

        outline_uniform = gl.glGetUniformLocation(program, b'outline');
        gl.glUniform1f(outline_uniform, self.outline);

        pixel_size_uniform = gl.glGetUniformLocation(program, b'pixel_size');
        gl.glUniform1f(pixel_size_uniform, camera.pixel_size);

        # bind everything and then draw the disks
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_position);
        gl.glEnableVertexAttribArray(0);
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, c_void_p(0));

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_mapcoord);
        gl.glEnableVertexAttribArray(1);
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, c_void_p(0));

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_color);
        gl.glEnableVertexAttribArray(2);
        gl.glVertexAttribPointer(2, 4, gl.GL_FLOAT, gl.GL_FALSE, 0, c_void_p(0));

        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.N*6);

        # unbind everything we bound
        gl.glDisableVertexAttribArray(0);
        gl.glDisableVertexAttribArray(1);
        gl.glDisableVertexAttribArray(2);
        gl.glUseProgram(0);

        # restore state
        gl.glPopAttrib(gl.GL_ENABLE_BIT | gl.GL_COLOR_BUFFER_BIT);

    ## Destroy OpenGL resources
    # OpenGL calls need to be made when a context is active. This class provides an explicit destroy() method so that
    # resources can be released at a controlled time. (not whenever python decides to call __del__.
    #
    def destroy(self):
        buf_list = numpy.array([self.buffer_position, self.buffer_mapcoord, self.buffer_color], dtype=numpy.uint32);
        gl.glDeleteBuffers(3, buf_list);

    ## Update the primitive with new values
    # \param prim base Primitive to represent
    # \param updated list of properties that were updated
    #
    def update(self, prim, updated):
        if any(prop in updated for prop in ['position', 'diameter', 'color']):
            position = numpy.zeros(shape=(self.N, 6, 2), dtype=numpy.float32);
            mapcoord = numpy.zeros(shape=(self.N, 6, 3), dtype=numpy.float32);
            color = numpy.zeros(shape=(self.N, 6, 4), dtype=numpy.float32);

            self.build_geometry(position, mapcoord, color, prim.positions,
                                prim.colors, prim.diameters);

            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_position);
            gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, None, position);

            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_mapcoord);
            gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, None, mapcoord);

            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_color);
            gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, None, color);

            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0);

## Triangle geometry
# \note GLTriangles is used internally by DrawGL and is not part of the public freud interface
#
# Store and draw the OpenGL geometry for the Triangles primitive.
#
# Triangles are drawn as-is. All vertices are specified directly in a triangle soup format. These are dumped into a
# buffer and passed to glDrawArrays().
#
# All variables are directly accessible class members.
#  - N: number of triangles
#  - buffer_vertices: OpenGL buffer (N*3 2-element positions)
#  - buffer_color: OpenGL buffer (N*3 4-element colors)
#
class GLTriangles(GLPrimitive):
    ## Vertex shader for drawing triangles
    #
    # Transform the incoming verts by the camera and pass through everything else.
    vertex_shader = """
#version 120

uniform mat4 camera;

attribute vec4 position;
attribute vec4 color;
attribute vec2 texcoord;

varying vec4 v_color;
varying vec2 v_texcoord;
void main()
    {
    gl_Position = camera * position;
    v_color = color;
    v_texcoord = texcoord;
    }
""";

    ## Fragment shader for drawing triangles
    #
    # Triangles are currently rendered in 2D mode, so no gamma correction is needed (colors are just passed through
    # directly).
    fragment_shader = """
#version 120

uniform int enable_tex;

uniform sampler2D tex;

varying vec4 v_color;
varying vec2 v_texcoord;

void main()
    {
    if ((enable_tex == 1) && !(v_color.r == 0.0f && v_color.g == 0.0f && v_color.b == 0.0f))
        {
        gl_FragColor = texture2D(tex, v_texcoord);
        }
    else
        {
        gl_FragColor = v_color;
        }
    }
""";

    ## Attributes for drawing disks
    attributes = ['position', 'color', 'texcoord'];

    ## Initialize a cached GL primitive
    # \param prim base Primitive to represent
    #
    def __init__(self, prim):
        GLPrimitive.__init__(self, prim);
        # simple scalar values
        self.N = int(len(prim.vertices));

        # initialize values for buffers
        color = numpy.zeros(shape=(self.N, 3, 4), dtype=numpy.float32);

        # start all coords at the center, with all the same color
        for i in range(3):
            color[:,i,:] = prim.colors;

        # generate OpenGL buffers and copy data
        self.buffer_position = gl.glGenBuffers(1);
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_position);
        gl.glBufferData(gl.GL_ARRAY_BUFFER, prim.vertices, gl.GL_STATIC_DRAW);

        self.buffer_color = gl.glGenBuffers(1);
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_color);
        gl.glBufferData(gl.GL_ARRAY_BUFFER, color, gl.GL_STATIC_DRAW);

        self.buffer_texcoord = gl.glGenBuffers(1);
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_texcoord);
        gl.glBufferData(gl.GL_ARRAY_BUFFER, prim.texcoords, gl.GL_STATIC_DRAW);

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0);

        if prim.tex_fname is not None:
            # load texture
            tex_img = QtGui.QImage(prim.tex_fname);
            tex_argb_img = tex_img.convertToFormat(QtGui.QImage.Format_ARGB32);
            img_data = numpy.array(tex_argb_img.constBits());

            # remap to RGBA
            rgba_data = numpy.zeros(shape=(tex_argb_img.width() * tex_argb_img.height(), 4), dtype=numpy.uint8);
            rgba_data = rgba_data.reshape((tex_img.width()*tex_img.height(), 4));
            img_data = img_data.reshape((tex_img.width()*tex_img.height(), 4));

            rgba_data[:,0] = img_data[:,2];
            rgba_data[:,1] = img_data[:,1];
            rgba_data[:,2] = img_data[:,0];
            rgba_data[:,3] = img_data[:,3];

            # setup texture object
            self.texture_object = gl.glGenTextures(1);
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_object);

            # Texture parameters are part of the texture object, so you need to
            # specify them only once for a given texture object.
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0);
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0);
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, tex_argb_img.width(), tex_argb_img.height(), 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, rgba_data);

            gl.glBindTexture(gl.GL_TEXTURE_2D, 0);
        else:
            self.texture_object = None;


    ## Draw the primitive
    # \param program OpenGL shader program
    # \param camera The camera to use when drawing
    #
    def draw(self, program, camera):
        # save state
        gl.glPushAttrib(gl.GL_ENABLE_BIT | gl.GL_COLOR_BUFFER_BIT);

        # setup state
        gl.glEnable(gl.GL_MULTISAMPLE);
        gl.glEnable(gl.GL_BLEND);
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA);

        gl.glUseProgram(program);

        # update the camera matrix
        camera_uniform = gl.glGetUniformLocation(program, b'camera');
        gl.glUniformMatrix4fv(camera_uniform, 1, True, camera.ortho_2d_matrix);

        # set whether textures are enabled
        enable_tex_uniform = gl.glGetUniformLocation(program, b'enable_tex');
        if self.texture_object is None:
            gl.glUniform1i(enable_tex_uniform, 0);
        else:
            gl.glUniform1i(enable_tex_uniform, 1);

            tex_unit = gl.glGetUniformLocation(program, b'tex');
            gl.glUniform1i(tex_unit, 0);
            gl.glActiveTexture(gl.GL_TEXTURE0);
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_object);

        # bind everything and then draw the disks
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_position);
        gl.glEnableVertexAttribArray(0);
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, c_void_p(0));

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_color);
        gl.glEnableVertexAttribArray(1);
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, 0, c_void_p(0));

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_texcoord);
        gl.glEnableVertexAttribArray(2);
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, c_void_p(0));

        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.N*3);

        # unbind everything we bound
        gl.glDisableVertexAttribArray(0);
        gl.glDisableVertexAttribArray(1);
        gl.glUseProgram(0);

        # restore state
        gl.glPopAttrib(gl.GL_ENABLE_BIT | gl.GL_COLOR_BUFFER_BIT);

    ## Destroy OpenGL resources
    # OpenGL calls need to be made when a context is active. This class provides an explicit destroy() method so that
    # resources can be released at a controlled time. (not whenever python decides to call __del__.
    #
    def destroy(self):
        buf_list = numpy.array([self.buffer_position, self.buffer_color], dtype=numpy.uint32);
        gl.glDeleteBuffers(2, buf_list);

## Rotated Triangle geometry
# \note GLPolygons is used internally by DrawGL and is not part of the public freud interface
#
# Store and draw the OpenGL geometry for the Triangles primitive.
#
# Triangles are drawn as-is. All vertices are specified directly in a triangle soup format. These are dumped into a
# buffer and passed to glDrawArrays().
#
# All variables are directly accessible class members.
#  - N: number of triangles
#  - buffer_vertices: OpenGL buffer (N*3 2-element positions)
#  - buffer_color: OpenGL buffer (N*3 4-element colors)
#
class GLPolygons(GLTriangles):
    ## Vertex shader for drawing triangles
    #
    # Transform the incoming verts by the camera and pass through everything else.
    vertex_shader = """
#version 120

uniform mat4 camera;

attribute vec4 position;
attribute float orientation;
attribute vec2 image;
attribute vec4 color;
attribute vec2 texcoord;

varying vec4 v_color;
varying vec2 v_texcoord;

void main()
    {
    float stheta = sin(orientation);
    float ctheta = cos(orientation);

    // rotate the image point into the correct orientation
    gl_Position.x = image.x*ctheta - image.y*stheta;
    gl_Position.y = image.x*stheta + image.y*ctheta;

    // shift into position
    gl_Position += position;

    // transform to screen coordinates
    gl_Position = camera * gl_Position;
    v_color = color;
    v_texcoord = texcoord;
    }
""";

    ## Attributes for drawing rotated triangles
    attributes = ['position', 'orientation', 'images', 'color', 'texcoords'];
    # buffer_position = None
    # buffer_orientation = None
    # buffer_image = None
    # buffer_color = None
    # buffer_texcoord = None

    ## Initialize a cached GL primitive
    # \param prim base Primitive to represent
    #
    def __init__(self, prim):
        GLPrimitive.__init__(self, prim);
        # simple scalar values
        self.N = int(len(prim.positions));

        # generate OpenGL buffers and copy data
        self.buffer_positions = gl.glGenBuffers(1);
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_positions);
        gl.glBufferData(gl.GL_ARRAY_BUFFER, prim.positions, gl.GL_STATIC_DRAW);

        self.buffer_orientations = gl.glGenBuffers(1);
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_orientations);
        gl.glBufferData(gl.GL_ARRAY_BUFFER, prim.orientations, gl.GL_STATIC_DRAW);

        self.buffer_images = gl.glGenBuffers(1);
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_images);
        gl.glBufferData(gl.GL_ARRAY_BUFFER, prim.images, gl.GL_STATIC_DRAW);

        self.buffer_colors = gl.glGenBuffers(1);
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_colors);
        gl.glBufferData(gl.GL_ARRAY_BUFFER, prim.colors, gl.GL_STATIC_DRAW);

        self.buffer_texcoords = gl.glGenBuffers(1);
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_texcoords);
        gl.glBufferData(gl.GL_ARRAY_BUFFER, prim.texcoords, gl.GL_STATIC_DRAW);

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0);

        if prim.tex_fname is not None:
            # load texture
            tex_img = QtGui.QImage(prim.tex_fname);
            tex_argb_img = tex_img.convertToFormat(QtGui.QImage.Format_ARGB32);
            img_data = numpy.array(tex_argb_img.constBits());

            # remap to RGBA
            rgba_data = numpy.zeros(shape=(tex_argb_img.width() * tex_argb_img.height(), 4), dtype=numpy.uint8);
            rgba_data = rgba_data.reshape((tex_img.width()*tex_img.height(), 4));
            img_data = img_data.reshape((tex_img.width()*tex_img.height(), 4));

            rgba_data[:,0] = img_data[:,2];
            rgba_data[:,1] = img_data[:,1];
            rgba_data[:,2] = img_data[:,0];
            rgba_data[:,3] = img_data[:,3];

            # setup texture object
            self.texture_object = gl.glGenTextures(1);
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_object);

            # Texture parameters are part of the texture object, so you need to
            # specify them only once for a given texture object.
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0);
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0);
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, tex_argb_img.width(), tex_argb_img.height(), 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, rgba_data);

            gl.glBindTexture(gl.GL_TEXTURE_2D, 0);
        else:
            self.texture_object = None;

    ## Draw the primitive
    # \param program OpenGL shader program
    # \param camera The camera to use when drawing
    #
    def draw(self, program, camera):
        # save state
        gl.glPushAttrib(gl.GL_ENABLE_BIT | gl.GL_COLOR_BUFFER_BIT);

        # setup state
        gl.glEnable(gl.GL_MULTISAMPLE);
        gl.glEnable(gl.GL_BLEND);
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA);

        gl.glUseProgram(program);

        # update the camera matrix
        camera_uniform = gl.glGetUniformLocation(program, b'camera');
        gl.glUniformMatrix4fv(camera_uniform, 1, True, camera.ortho_2d_matrix);

        # set whether textures are enabled
        enable_tex_uniform = gl.glGetUniformLocation(program, b'enable_tex');
        if self.texture_object is None:
            gl.glUniform1i(enable_tex_uniform, 0);
        else:
            gl.glUniform1i(enable_tex_uniform, 1);

            tex_unit = gl.glGetUniformLocation(program, b'tex');
            gl.glUniform1i(tex_unit, 0);
            gl.glActiveTexture(gl.GL_TEXTURE0);
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_object);

        # bind everything and then draw the triangles
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_positions);
        gl.glEnableVertexAttribArray(0);
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, c_void_p(0));

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_orientations);
        gl.glEnableVertexAttribArray(0);
        gl.glVertexAttribPointer(1, 1, gl.GL_FLOAT, gl.GL_FALSE, 0, c_void_p(0));

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_images);
        gl.glEnableVertexAttribArray(2);
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, c_void_p(0));

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_colors);
        gl.glEnableVertexAttribArray(3);
        gl.glVertexAttribPointer(3, 4, gl.GL_FLOAT, gl.GL_FALSE, 0, c_void_p(0));

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_texcoords);
        gl.glEnableVertexAttribArray(4);
        gl.glVertexAttribPointer(4, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, c_void_p(0));

        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.N*3);

        # unbind everything we bound
        gl.glDisableVertexAttribArray(0);
        gl.glDisableVertexAttribArray(1);
        gl.glUseProgram(0);

        # restore state
        gl.glPopAttrib(gl.GL_ENABLE_BIT | gl.GL_COLOR_BUFFER_BIT);

    ## Destroy OpenGL resources
    # OpenGL calls need to be made when a context is active. This class provides an explicit destroy() method so that
    # resources can be released at a controlled time. (not whenever python decides to call __del__.
    #
    def destroy(self):
        buf_list = numpy.array([self.buffer_positions, self.buffer_orientations,
                                self.buffer_images, self.buffer_colors,
                                self.buffer_texcoords], dtype=numpy.uint32);
        gl.glDeleteBuffers(5, buf_list);

    ## Update the primitive with new values
    # \param prim base Primitive to represent
    # \param updated list of properties that were updated
    #
    def update(self, prim, updated):

        for prop in updated:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, getattr(self, 'buffer_{}'.format(prop)));
            gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, None, getattr(prim, prop));

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0);

        if prim.tex_fname is not None:
            # load texture
            tex_img = QtGui.QImage(prim.tex_fname);
            tex_argb_img = tex_img.convertToFormat(QtGui.QImage.Format_ARGB32);
            img_data = numpy.array(tex_argb_img.constBits());

            # remap to RGBA
            rgba_data = numpy.zeros(shape=(tex_argb_img.width() * tex_argb_img.height(), 4), dtype=numpy.uint8);
            rgba_data = rgba_data.reshape((tex_img.width()*tex_img.height(), 4));
            img_data = img_data.reshape((tex_img.width()*tex_img.height(), 4));

            rgba_data[:,0] = img_data[:,2];
            rgba_data[:,1] = img_data[:,1];
            rgba_data[:,2] = img_data[:,0];
            rgba_data[:,3] = img_data[:,3];

            # setup texture object
            self.texture_object = gl.glGenTextures(1);
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_object);

            # Texture parameters are part of the texture object, so you need to
            # specify them only once for a given texture object.
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0);
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0);
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, tex_argb_img.width(), tex_argb_img.height(), 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, rgba_data);

            gl.glBindTexture(gl.GL_TEXTURE_2D, 0);
        else:
            self.texture_object = None;
