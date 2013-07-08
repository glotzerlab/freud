from __future__ import division, print_function
import time
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

from freud.util import triangulate
from freud.util import trimath

import _freud;

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
            self.colors = numpy.array(colors, dtype=numpy.float32);

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
    def __init__(self, vertices, texcoords=None, colors=None, color=None, tex_fname=None):
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

        if texcoords is None:
            self.texcoords = numpy.zeros(shape=self.vertices.shape(), dtype=numpy.float32)
        else:
            self.texcoords = numpy.array(texcoords, dtype=numpy.float32)

        if len(self.texcoords.shape) != 3:
            raise TypeError("texcoords must be a Nx3x2 array");
        if self.texcoords.shape[1] != 3:
            raise ValueError("texcoords must be a Nx3x2 array");
        if self.texcoords.shape[2] != 2:
            raise ValueError("texcoords must be a Nx3x2 array");

        self.tex_fname = tex_fname

        # -----------------------------------------------------------------
        # set up colors
        if colors is None:
            self.colors = numpy.zeros(shape=(N,4), dtype=numpy.float32);
            self.colors[:,3] = 1;
        else:
            self.colors = numpy.array(colors, dtype=numpy.float32);

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
    def __init__(self, positions, angles, polygon, colors=None, color=None, outline=0.1, tex_fname=None):
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
        self.angles = numpy.array(angles, dtype=numpy.float32);

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
            self.colors = numpy.array(colors, dtype=numpy.float32);

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
        # create a triangulation class
        tmp_poly = triangulate.triangulate(polygon, outline)
        # decompose the polygon into constituent triangles
        tmp_poly.calculate()
        # put the triangle vertices into a numpy array
        triangle_array = tmp_poly.getTriangles()
        textriangle_array = tmp_poly.getTexTriangles()
        outline_array = tmp_poly.getOutline()
        N_T = triangle_array.shape[0]
        N_O = outline_array.shape[0]

        # This is slow
        # vert_array = numpy.zeros(shape=tuple([N * N_T, 3, 2]), dtype=numpy.float32)
        # color_array = numpy.zeros(shape=tuple([N * N_T, 4]), dtype=numpy.float32)
        # positions_array = self.positions
        # angles_array = self.angles
        # poly_color_array = self.colors

        # _freud.triangle_rotate(vert_array, color_array, positions_array, angles_array, triangle_array, poly_color_array)

        vert_array = numpy.zeros(shape=tuple([N * N_T, 3, 2]), dtype=numpy.float32)
        color_array = numpy.zeros(shape=tuple([N * N_T, 4]), dtype=numpy.float32)
        overt_array = numpy.zeros(shape=tuple([N * N_O, 3, 2]), dtype=numpy.float32)
        ocolor_array = numpy.zeros(shape=tuple([N * N_O, 4]), dtype=numpy.float32)
        tex_array = numpy.zeros(shape=tuple([N * N_T, 3, 2]), dtype=numpy.float32)
        tex_color_array = numpy.zeros(shape=tuple([N * N_T, 4]), dtype=numpy.float32)
        positions_array = self.positions
        dummy_positions = numpy.zeros(shape=self.positions.shape, dtype=numpy.float32)
        dummy_angles = numpy.zeros(shape=self.angles.shape, dtype=numpy.float32)
        angles_array = self.angles
        poly_color_array = self.colors
        out_color_array = numpy.zeros(shape=tuple([N, 4]), dtype=numpy.float32)
        out_color_array[:,:] = numpy.array([0.0, 0.0, 0.0, 1.0], dtype=numpy.float32)
# Need to broadcast the N_T tex coords into a N*N_T array...would rather not have to do it in C...
        _freud.triangle_rotate_mat(vert_array, color_array, positions_array, angles_array, triangle_array, poly_color_array)
        _freud.triangle_rotate_mat(overt_array, ocolor_array, positions_array, angles_array, outline_array, out_color_array)
        _freud.triangle_rotate_mat(tex_array, tex_color_array, dummy_positions, dummy_angles, textriangle_array, poly_color_array)
        vert_array = numpy.concatenate([vert_array, overt_array])
        color_array = numpy.concatenate([color_array, ocolor_array])
        # -----------------------------------------------------------------
        if tex_fname is not None:
            Triangles.__init__(self, vert_array, texcoords = tex_array, colors = color_array, tex_fname = tex_fname);
        else:
            Triangles.__init__(self, vert_array, colors = color_array);


class TexturedPolygons(Triangles):
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
    def __init__(self, positions, angles, polygon, colors=None, color=None, outline=0.1, tex=False, tex_file=None):
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
        self.angles = numpy.array(angles, dtype=numpy.float32);

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
            self.colors = numpy.array(colors, dtype=numpy.float32);

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
        # create a triangulation class
        tmp_poly = triangulate.triangulate(polygon, outline)
        # decompose the polygon into constituent triangles
        tmp_poly.calculate()
        # put the triangle vertices into a numpy array
        triangle_array = tmp_poly.getTriangles()
        outline_array = tmp_poly.getOutline()
        N_T = triangle_array.shape[0]
        N_O = outline_array.shape[0]

        # This is slow
        # vert_array = numpy.zeros(shape=tuple([N * N_T, 3, 2]), dtype=numpy.float32)
        # color_array = numpy.zeros(shape=tuple([N * N_T, 4]), dtype=numpy.float32)
        # positions_array = self.positions
        # angles_array = self.angles
        # poly_color_array = self.colors

        # _freud.triangle_rotate(vert_array, color_array, positions_array, angles_array, triangle_array, poly_color_array)

        vert_array = numpy.zeros(shape=tuple([N * N_T, 3, 2]), dtype=numpy.float32)
        color_array = numpy.zeros(shape=tuple([N * N_T, 4]), dtype=numpy.float32)
        overt_array = numpy.zeros(shape=tuple([N * N_O, 3, 2]), dtype=numpy.float32)
        ocolor_array = numpy.zeros(shape=tuple([N * N_O, 4]), dtype=numpy.float32)
        positions_array = self.positions
        angles_array = self.angles
        poly_color_array = self.colors
        out_color_array = numpy.zeros(shape=tuple([N, 4]), dtype=numpy.float32)
        out_color_array[:,:] = numpy.array([0.0, 0.0, 0.0, 1.0], dtype=numpy.float32)

        _freud.triangle_rotate_mat(vert_array, color_array, positions_array, angles_array, triangle_array, poly_color_array)
        _freud.triangle_rotate_mat(overt_array, ocolor_array, positions_array, angles_array, outline_array, out_color_array)
        vert_array = numpy.concatenate([vert_array, overt_array])
        color_array = numpy.concatenate([color_array, ocolor_array])
        # -----------------------------------------------------------------
        # set up outline
        Triangles.__init__(self, vert_array, N_T, N_O, tex_file, colors = color_array);
        #self.outline = outline;
            # img = Image.open(tex_file)

## Approximated Spheropolygons
#
# Represent N instances of a spheropolygon in 2D. For this, the
# rounded edges have been approximated by a given number of points on
# the curve. Black edges are drawn given a global outline width.
class Spheropolygons(RepeatedPolygons):
    ## Initialize a spheropolygon primitive
    # \param positions Nx2 array listing the positions of each polygon (in distance units)
    # \param angles N array listing the rotation of the polygon about its center (in radians)
    # \param polygon Kx2 array listing the coordinates of each polygon in its local frame (in distance units)
    # \param colors Nx4 array listing the colors (rgba 0.0-1.0) of each polygon (in SRGB)
    # \param color 4 element iterable listing the color to be applied to every polygon (in SRGB)
    #              \a color overrides anything set by colors
    # \param outline Outline width (in distance units)
    # \param radius Radius of the disk to round by
    # \param granularity Number of points to place on each curved edge
    #
    # When colors is none, it defaults to (0,0,0,1) for each particle.
    #
    # \note N **must** be the same for each array
    #
    # After initialization, the instance will have members positions, angles, polygon and colors, each being a numpy
    # array of the appropriate size and dtype float32. Users should not modify these directly, they are intended for
    # use only by renderers. Instead, users should create a new primitive from scratch to rebuild geometry.
    #
    def __init__(self, positions, angles, polygon, colors=None, color=None, outline=0.1, radius=1.0, granularity=5):
        polygon = self.roundCorners(polygon, radius, granularity)

        super(Spheropolygons, self).__init__(positions, angles, polygon, colors, color, outline)

    ## Round a polygon by a given radius
    # \param vertices Nx2 array or list-like object of points in the polygon
    # \param radius Radius of the disk to round by
    # \param granularity Number of points to place on each curved edge
    #
    # Returns a list of vertices which approximately enlarges the
    # input polygon by a disk of the given radius. Assumes vertices
    # are given in counter-clockwise order.
    #
    def roundCorners(self, vertices, radius=1.0, granularity=5):
        # Make 3D unit vectors drs from each vertex i to its neighbor i+1
        vertices = numpy.array(vertices)
        drs = numpy.roll(vertices, -1, axis=0) - vertices
        drs /= numpy.sqrt(numpy.sum(drs*drs, axis=1))[:, numpy.newaxis]
        drs = numpy.hstack([drs, numpy.zeros((drs.shape[0], 1))])

        # relStarts and relEnds are the offsets relative to the first and
        # second point of each line segment in the polygon.
        rvec = numpy.array([[0, 0, -1]])*radius
        relStarts = numpy.cross(rvec, drs)[:, :2]
        relEnds =  numpy.cross(rvec, drs)[:, :2]

        # absStarts and absEnds are the beginning and end points for each
        # straight line segment.
        absStarts = vertices + relStarts
        absEnds = numpy.roll(vertices, -1, axis=0) + relEnds

        relStarts = numpy.roll(relStarts, -1, axis=0)

        # We will join each of these segments by a round cap; this will be
        # done by tracing an arc with the given radius, centered at each
        # vertex from an end of a line segment to a beginning of the next
        theta1s = numpy.arctan2(relEnds[:, 1], relEnds[:, 0])
        theta2s = numpy.arctan2(relStarts[:, 1], relStarts[:, 0])
        dthetas = (theta2s - theta1s) % (2*numpy.pi)

        # thetas are the angles at which we'll place points for each
        # vertex; curves are the points on the approximate curves on the
        # corners.
        thetas = numpy.zeros((vertices.shape[0], granularity))
        for i, (theta1, dtheta) in enumerate(zip(theta1s, dthetas)):
            thetas[i] = theta1 + numpy.linspace(0, dtheta, 2 + granularity)[1:-1]
        curves = radius*numpy.vstack([numpy.cos(thetas).flat, numpy.sin(thetas).flat]).T
        curves = curves.reshape((-1, granularity, 2))
        curves += numpy.roll(vertices, -1, axis=0)[:, numpy.newaxis, :]

        # Now interleave the pieces
        result = []
        for (end, curve, start, vert, dtheta) in zip(absEnds, curves,
                                                     numpy.roll(absStarts, -1, axis=0),
                                                     numpy.roll(vertices, -1, axis=0),
                                                     dthetas):
            # convex case: add the end of the last straight line
            # segment, the curved edge, then the start of the next
            # straight line segment.
            if dtheta <= numpy.pi:
                result.append(end)
                result.append(curve)
                result.append(start)
            # concave case: don't use the curved region, just find the
            # intersection and add that point.
            else:
                l = radius/numpy.cos(dtheta/2)
                p = 2*vert - start - end
                p /= trimath.norm(p)
                result.append(vert + p*l)

        result = numpy.vstack(result)

        return result

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
