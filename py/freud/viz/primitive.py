from __future__ import division, print_function
import math
import numpy
import logging
logger = logging.getLogger(__name__);

try:
    from PySide import QtGui
except ImportError:
    QtGui = None;
    logger.info('PySide is not available, Image saving is disabled');

from freud.viz import base
from freud.viz import colorutil

## \package freud.viz.primitive
#
# Definition of basic viz primitives
#

## Disk primitive (2D)
#
# Represent N disks in 2D. Each has a given color and a global outline width is specified.
class Disks(base.Primitive):
    ## Initialize a disk primitive
    # \param positions Nx2 array listing the positions of each disk (in distance units)
    # \param diameters N array listing the diameters of each disk (in distance units)
    # \param colors Nx4 array listing the colors (rgba 0.0-1.0) of each disk (in SRGB)
    # \param color 4 element iterable listing the color to be applied to every disk. (in SRGB)
    #              \a color overrides anything set by colors
    # \param outline Outline width (in distance units)
    #
    # When \a diameters is None, it defaults to 1.0 for each particle. When colors is none, it defaults to 
    # (0,0,0,1) for each particle.
    #
    # \note N **must** be the same for each array
    #
    # After initialization, the instance will have members positions, diameters, and colors, each being a numpy
    # array of the appropriate size and dtype float32. Users should not modify these directly, they are intended for
    # use only by renderers. Instead, users should create a new primitive from scratch to rebuild geometry.
    #
    def __init__(self, positions, diameters=None, colors=None, color=None, outline=0.1):
        base.Primitive.__init__(self);
        
        # -----------------------------------------------------------------
        # set up positions
        # convert to a numpy array
        self.positions = numpy.array(positions, dtype=numpy.float32);
        # error check the input
        if len(self.positions.shape) != 2:
            raise TypeError("positions must be a Nx2 array");
        if self.positions.shape[1] != 2:
            raise ValueError("positions must be a Nx2 array");
        
        N = self.positions.shape[0];
        
        # -----------------------------------------------------------------
        # set up diameters
        if diameters is None:
            self.diameters = numpy.zeros(shape=(N,), dtype=numpy.float32);
            self.diameters[:] = 1;
        else:
            self.diameters = numpy.array(diameters);
        
        # error check diameters
        if len(self.diameters.shape) != 1:
            raise TypeError("diameters must be a single dimension array");
        if self.diameters.shape[0] != N:
            raise ValueError("diameters must have N the same as positions");
        
        # -----------------------------------------------------------------
        # set up colors
        if colors is None:
            self.colors = numpy.zeros(shape=(N,4), dtype=numpy.float32);
            self.colors[:,3] = 1;
        else:
            self.colors = numpy.array(colors);
        
        # error check colors
        if len(self.colors.shape) != 2:
            raise TypeError("colors must be a Nx4 array");
        if self.colors.shape[1] != 4:
            raise ValueError("colors must have N the same as positions");
        if self.colors.shape[0] != N:
            raise ValueError("colors must have N the same as positions");
        
        if color is not None:
            acolor = numpy.array(color);
            if len(acolor.shape) != 1:
                raise TypeError("color must be a 4 element array");
            if acolor.shape[0] != 4:
                raise ValueError("color must be a 4 element array");

            self.colors[:,:] = acolor;
    
        # -----------------------------------------------------------------
        # set up outline
        self.outline = outline;

## Line primitive
#
# Represent N lines in 2D or 3D (2D specific renderers may simply ignore the z component).
class Lines(base.Primitive):
    pass


## Triangle primitive
#
# Represent N triangles in 2D, each defined by vertices and a color.
#
class Triangles(base.Primitive):
    ## Initialize a disk primitive
    # \param vertices Nx3x2 array listing the vertices of each triangle (in distance units)
    # \param colors Nx4 array listing the colors (rgba 0.0-1.0) of each triangle (in SRGB)
    # \param color 4 element iterable listing the color to be applied to every triangle (in SRGB)
    #              \a color overrides anything set by colors
    #
    # When colors is none, it defaults to (0,0,0,1) for each particle.
    #
    # \note N **must** be the same for each array
    #
    # After initialization, the instance will have members vertices and colors, each being a numpy
    # array of the appropriate size and dtype float32. Users should not modify these directly, they are intended for
    # use only by renderers. Instead, users should create a new primitive from scratch to rebuild geometry.
    #
    def __init__(self, vertices, colors=None, color=None):
        base.Primitive.__init__(self);
        
        # -----------------------------------------------------------------
        # set up vertices
        # convert to a numpy array
        self.vertices = numpy.array(vertices, dtype=numpy.float32);        
        # error check the input
        if len(self.vertices.shape) != 3:
            raise TypeError("vertices must be a Nx3x2 array");
        if self.vertices.shape[1] != 3:
            raise ValueError("vertices must be a Nx3x2 array");
        if self.vertices.shape[2] != 2:
            raise ValueError("vertices must be a Nx3x2 array");

        N = self.vertices.shape[0];
        
        # -----------------------------------------------------------------
        # set up colors
        if colors is None:
            self.colors = numpy.zeros(shape=(N,4), dtype=numpy.float32);
            self.colors[:,3] = 1;
        else:
            self.colors = numpy.array(colors);
        
        # error check colors
        if len(self.colors.shape) != 2:
            raise TypeError("colors must be a Nx4 array");
        if self.colors.shape[1] != 4:
            raise ValueError("colors must be a Nx4 array");
        if self.colors.shape[0] != N:
            raise ValueError("colors must have N the same as positions");
        
        if color is not None:
            acolor = numpy.array(color);
            if len(acolor.shape) != 1:
                raise TypeError("color must be a 4 element array");
            if acolor.shape[0] != 4:
                raise ValueError("color must be a 4 element array");

            self.colors[:,:] = acolor;

## Repeated polygons
#
# Represent N instances of the same polygon in 2D, each at a different position, orientation, and color. Black edges
# are drawn given a global outline width.
#
class RepeatedPolygons(Triangles):
    ## Initialize a disk primitive
    # \param positions Nx2 array listing the positions of each polygon (in distance units)
    # \param angles N array listing the rotation of the polygon about its center (in radians)
    # \param polygon Kx2 array listing the coordinates of each polygon in its local frame (in distance units)
    # \param colors Nx4 array listing the colors (rgba 0.0-1.0) of each polygon (in SRGB)
    # \param color 4 element iterable listing the color to be applied to every polygon (in SRGB)
    #              \a color overrides anything set by colors
    # \param outline Outline width (in distance units)
    #
    # When colors is none, it defaults to (0,0,0,1) for each particle.
    #
    # \note N **must** be the same for each array
    #
    # After initialization, the instance will have members positions, angles, polygon and colors, each being a numpy
    # array of the appropriate size and dtype float32. Users should not modify these directly, they are intended for
    # use only by renderers. Instead, users should create a new primitive from scratch to rebuild geometry.
    #
    def __init__(self, positions, angles, polygon, colors=None, color=None, outline=0.1):   
        
        from freud.util import triangulate
        from freud.util import trimath
        
        # -----------------------------------------------------------------
        # set up positions
        # convert to a numpy array
        self.positions = numpy.array(positions, dtype=numpy.float32);        
        # error check the input
        if len(self.positions.shape) != 2:
            raise TypeError("positions must be a Nx2 array");
        if self.positions.shape[1] != 2:
            raise ValueError("positions must be a Nx2 array");
        
        N = self.positions.shape[0];
        
        # -----------------------------------------------------------------
        # set up angles
        self.angles = numpy.array(angles);
        
        # error check angles
        if len(self.angles.shape) != 1:
            raise TypeError("angles must be a single dimension array");
        if self.angles.shape[0] != N:
            raise ValueError("angles must have N the same as positions");
        
        # -----------------------------------------------------------------
        # set up polygon
        self.polygon = numpy.array(polygon);
        
        # error check polygon
        if len(self.polygon.shape) != 2:
            raise TypeError("polygon must be a Kx2 array");
        if self.polygon.shape[0] < 3:
            raise ValueError("polygon must have at least 3 vertices");
        if self.polygon.shape[1] < 2:
            raise ValueError("polygon must be a Kx2 array");

        # -----------------------------------------------------------------
        # set up colors
        if colors is None:
            self.colors = numpy.zeros(shape=(N,4), dtype=numpy.float32);
            self.colors[:,3] = 1;
        else:
            self.colors = numpy.array(colors);
        
        # error check colors
        if len(self.colors.shape) != 2:
            raise TypeError("colors must be a Nx4 array");
        if self.colors.shape[1] != 4:
            raise ValueError("colors must have N the same as positions");
        if self.colors.shape[0] != N:
            raise ValueError("colors must have N the same as positions");
        
        if color is not None:
            acolor = numpy.array(color);
            if len(acolor.shape) != 1:
                raise TypeError("color must be a 4 element array");
            if acolor.shape[0] != 4:
                raise ValueError("color must be a 4 element array");

            self.colors[:,:] = acolor;
        
        tmp_poly = triangulate.triangulate(polygon)
        tmp_poly.calculate()
        t_verts = numpy.array(tmp_poly.getTriangles())
        N_T = t_verts.shape[0]
        #t_verts = tmp_poly.getTriangles()
        #print(t_verts)
        
        # Need to take the triangle array and use it to populate
        # A list for all particles
        # Create the list of verts
        vert_array = []
        for i in range(N):
            poly_plain_verts = []
            for j in range(N_T):
                rot_t = trimath.tri_rotate(t_verts[j], self.angles[i])
                trans_t = rot_t + self.positions[i]
                plain_t_arr = []
                for k in range(3):
                    plain_t_arr.append((trans_t[k][0], trans_t[k][1]))
                vert_array.append(plain_t_arr)
        vert_array = numpy.array(vert_array)
        #print(vert_array)
            # Need to rotate and move
        
        # -----------------------------------------------------------------
        # set up outline
        Triangles.__init__(self, vert_array);
        #self.outline = outline;

## Image
#
# The Image primitive takes in a 2D array of sRGBA values and draws that image at the specified point and size in
# space. It is useful for overlaying heat maps on top of simulation data.
#
class Image(base.Primitive):
    ## Initialize a image primitive
    # \param position 2-element array containing the position of the image (in distance units)
    # \param size 2-element array containing the size of the image (in distance units)
    # \param data NxMx4 array containing the image data in sRGBA 0.0-1.0 format
    # \param filename (if specified) A filename (must end in .png) to save the image to
    #
    # After initialization, the instance will have members position, size, and data, each being a numpy
    # array of the appropriate size and dtype float32. Users should not modify these directly, they are intended for
    # use only by renderers. Instead, users should create a new primitive from scratch to rebuild geometry.
    #
    # \a filename may be used by renderers that need to save images out for input into other programs. It can be used 
    # to control what filename the image is saved as. Users can also trigger a save by calling save(). \a filename is 
    # relative to the current working directory at the time save() is called (or at the time a renderer's write() is 
    # called). If no filename is specified, the renderer will auto-generate one as needed.
    #
    def __init__(self, position, size, data, filename=None):
        base.Primitive.__init__(self);
        
        # -----------------------------------------------------------------
        # set up position
        # convert to a numpy array
        self.position = numpy.array(position, dtype=numpy.float32);        
        # error check the input
        if len(self.position.shape) != 1:
            raise TypeError("position must be a 2-element array");
        if self.position.shape[0] != 2:
            raise ValueError("position must be a 2-element array");
        
        # -----------------------------------------------------------------
        # set up size
        # convert to a numpy array
        self.size = numpy.array(size, dtype=numpy.float32);        
        # error check the input
        if len(self.size.shape) != 1:
            raise TypeError("size must be a 2-element array");
        if self.size.shape[0] != 2:
            raise ValueError("size must be a 2-element array");
        
        # -----------------------------------------------------------------
        # set up data
        # convert to a numpy array
        self.data = numpy.array(data, dtype=numpy.float32);        
        # error check the input
        if len(self.data.shape) != 3:
            raise TypeError("data must be a NxMx4 array");
        if self.data.shape[2] != 4:
            raise ValueError("data must be a NxMx4 array");

        self.filename = filename;
        
    ## Save the image to a png file
    # \param filename Filename to save to (must end in .png)
    #
    # Save the image data to a png file
    #
    # \note Image saving requires PySide (an optional dependency). If PySide is not available, save will raise an
    # exception.
    #
    def save(self, filename):
        if QtGui is None:
            raise RuntimeError("save requires PySide");
        
        # convert to ARGB32 format and init QImage
        data_bytes = colorutil.sRGBAtoARGB32(self.data);
        img = QtGui.QImage(data_bytes, data_bytes.shape[0], data_bytes.shape[1], QtGui.QImage.Format_ARGB32);
        
        # save
        img.save(filename);
