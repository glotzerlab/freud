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
from freud.util.shapes import Polygon, Outline

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
            self.texcoords = numpy.zeros(shape=self.vertices.shape, dtype=numpy.float32)
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


## Rotated Triangle primitive
#
# Represent N shapes in 2D specified by positions, orientations, a set
# of triangles with local vertex images, and a color.
#
class RotatedTriangles(base.Primitive):
    ## Initialize a rotated triangle primitive
    # \param image Ntx3x2 array listing the local vertices of a shape
    # \param positions Npx2 array listing the positions of each particle
    # \param orientations Np-length array listing the orientation angle of each shape
    # \param colors Npx4 array listing the colors (rgba 0.0-1.0) of each shape (in SRGB)
    # \param color 4 element iterable listing the color to be applied to every triangle (in SRGB)
    #              \a color overrides anything set by colors
    #
    # When colors is none, it defaults to (0,0,0,1) for each particle.
    #
    # \note Np **must** be the same for each array
    #
    # After initialization, the instance will have members positions,
    # orientations, images, and colors, each being a numpy array of
    # the appropriate size and dtype float32. Users should not modify
    # these directly, they are intended for use only by
    # renderers. Instead, users should create a new primitive from
    # scratch to rebuild geometry.
    #
    def __init__(self, image, positions, orientations, *args, **kwargs):
        base.Primitive.__init__(self);
        self.update(image, positions, orientations, *args, **kwargs);

    def update(self, image, positions, orientations, texcoords=None, colors=None, color=None, tex_fname=None):

        # -----------------------------------------------------------------
        # set up image
        # convert to a numpy array
        self.image = numpy.array(image, dtype=numpy.float32);
        # error check the input
        if len(self.image.shape) != 3:
            raise TypeError("image must be a Ntx3x2 array");
        if self.image.shape[1] != 3:
            raise ValueError("image must be a Ntx3x2 array");
        if self.image.shape[2] != 2:
            raise ValueError("image must be a Ntx3x2 array");

        Nt = self.image.shape[0];

        # -----------------------------------------------------------------
        # set up positions
        # convert to a numpy array
        self.positions = numpy.array(positions, dtype=numpy.float32);
        # error check the input
        if len(self.positions.shape) != 2:
            raise TypeError("positions must be a Npx2 array");
        if self.positions.shape[1] != 2:
            raise ValueError("positions must be a Npx2 array");

        Np = self.positions.shape[0];

        # -----------------------------------------------------------------
        # set up orientations
        # convert to a numpy array
        self.orientations = numpy.array(orientations, dtype=numpy.float32);
        # error check the input
        if len(self.orientations.shape) != 1:
            raise TypeError("orientations must be a Np-length array");
        if len(self.orientations) != Np:
            raise ValueError("Must have the same number of orientations as positions");

        if texcoords is None:
            self.texcoords = numpy.zeros(shape=(3*Np*Nt, 2), dtype=numpy.float32)
        else:
            self.texcoords = numpy.array(texcoords, dtype=numpy.float32)

        # TODO: fix logic here
        # if len(self.texcoords.shape) != 3:
        #     raise TypeError("texcoords must be a Nx3x2 array");
        # if self.texcoords.shape[1] != 3:
        #     raise ValueError("texcoords must be a Nx3x2 array");
        # if self.texcoords.shape[2] != 2:
            # raise ValueError("texcoords must be a Nx3x2 array");

        self.tex_fname = tex_fname

        # -----------------------------------------------------------------
        # set up colors
        if colors is None:
            self.colors = numpy.zeros(shape=(3*Np*Nt,4), dtype=numpy.float32);
            self.colors[:,3] = 1;
        else:
            self.colors = numpy.array(colors, dtype=numpy.float32);

        # error check colors
        if len(self.colors.shape) != 2:
            raise TypeError("colors must be a Npx4 array");
        if self.colors.shape[1] != 4:
            raise ValueError("colors must be a Npx4 array");
        if self.colors.shape[0] != Np:
            raise ValueError("colors must have N the same as positions");

        if color is not None:
            acolor = numpy.array(color);
            if len(acolor.shape) != 1:
                raise TypeError("color must be a 4 element array");
            if acolor.shape[0] != 4:
                raise ValueError("color must be a 4 element array");

            self.colors[:,:] = acolor;

        # -----------------------------------------------------------------
        # broadcast data into the correct form
        finalSize = lambda k: (3*Np*Nt, k)

        self.images = numpy.tile(self.image, (Np, 1, 1)).reshape(finalSize(2))
        self.positions = numpy.tile(
            self.positions[:, numpy.newaxis, :], (1, 3*Nt, 1)).reshape(finalSize(2))
        self.colors = numpy.tile(
            self.colors[:, numpy.newaxis, :], (1, 3*Nt, 1)).reshape(finalSize(4))
        self.orientations = numpy.tile(
            self.orientations[:, numpy.newaxis], (1, 3*Nt)).reshape(finalSize(1))
        self.updated = ['images', 'positions', 'colors', 'orientations']


## Arrows
#
# Represent N arrows in 2D with different positions, orientations,
# lengths, line widths, and colors.
class Arrows(Triangles):
    ## Initialize an arrow primitive
    # \param positions Nx2 array listing the origin position of each arrow (in distance units)
    # \param widths N array or single value listing the line width of each arrow
    # \param lengths N array or single value listing the length of each arrow
    # \param angles N array or single value listing the orientation of each arrow (in radians)
    # \param colors Nx4 array listing the colors (rgba 0.0-1.0) of each polygon (in SRGB)
    # \param color 4 element iterable listing the color to be applied to every polygon (in SRGB)
    #              \a color overrides anything set by colors
    # \param aspectRatio Real length of a basic arrow of virtual length 1
    #
    # When colors is None, it defaults to (0,0,0,1) for each particle.
    #
    # \note N **must** be the same for each array
    #
    # After initialization, the instance will have members positions,
    # widths, lengths, angles, and colors, each being a numpy array of
    # the appropriate size and dtype float32. Users should not modify
    # these directly, they are intended for use only by
    # renderers. Instead, users should create a new primitive from
    # scratch to rebuild geometry.
    def __init__(self, positions, widths, lengths, angles, colors=None, color=None, aspectRatio=5.):
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
        # set up widths
        self.widths = numpy.array(widths, dtype=numpy.float32);
        if len(self.widths.shape) == 0:
            self.widths = numpy.repeat(self.widths, N);

        # error check widths
        if len(self.widths.shape) != 1:
            raise TypeError("widths must be a scalar or single dimension array");
        if self.widths.shape[0] != N:
            raise ValueError("widths must have N the same as positions or 1");

        # -----------------------------------------------------------------
        # set up lengths
        self.lengths = numpy.array(lengths, dtype=numpy.float32);
        if len(self.lengths.shape) == 0:
            self.lengths = numpy.repeat(self.lengths, N);

        # error check lengths
        if len(self.lengths.shape) != 1:
            raise TypeError("lengths must be a scalar or single dimension array");
        if self.lengths.shape[0] != N:
            raise ValueError("lengths must have N the same as positions or 1");

        # -----------------------------------------------------------------
        # set up angles
        self.angles = numpy.array(angles, dtype=numpy.float32);

        # error check angles
        if len(self.angles.shape) not in [0, 1]:
            raise TypeError("angles must be a scalar or single dimension array");
        if len(self.angles.shape) and self.angles.shape[0] != N:
            raise ValueError("angles must have N the same as positions or 1");

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

        # stem0 and stem1 are the two triangles for the rectangular
        # "stem" of the arrow
        stem0 = numpy.array([[[0, -.5], [0, .5], [aspectRatio, .5]]],
                            dtype=numpy.float32);
        stem1 = numpy.array([[[0, -.5], [aspectRatio, .5], [aspectRatio, -.5]]],
                            dtype=numpy.float32);
        # equilateral triangle tip
        # l is the height of an equilateral triangle of side length 3
        l = 1.5*numpy.sqrt(3);
        tip = numpy.array([[[0, 1.5], [l, 0], [0, -1.5]]], dtype=numpy.float32);

        stem0 = numpy.repeat(stem0, N, axis=0);
        stem1 = numpy.repeat(stem1, N, axis=0);
        tip = numpy.repeat(tip, N, axis=0);

        # scale the width of stem and the size of the tip by the given line widths
        stem0[:, :, 1] *= self.widths[:, numpy.newaxis];
        stem1[:, :, 1] *= self.widths[:, numpy.newaxis];
        tip *= self.widths[:, numpy.newaxis, numpy.newaxis];

        # scale the length of stem by the given line lengths
        stem0[:, 2, 0] *= self.lengths;
        stem1[:, 1:, 0] *= self.lengths[:, numpy.newaxis];

        # shrink the stem to just touch the tip
        stem0[:, 2, 0] -= tip[:, 2, 0];
        stem1[:, 1:, 0] -= tip[:, 2, 0][:, numpy.newaxis];

        # shift the tip to the end of the stem
        delta = aspectRatio*self.lengths - tip[:, 2, 0];
        tip[:, :, 0] += delta[:, numpy.newaxis];

        # rotate the vertices into the correct orientation
        rmat = numpy.empty((N, 2, 2));
        rmat[:, 0, 0] = rmat[:, 1, 1] = numpy.cos(self.angles);
        rmat[:, 1, 0] = numpy.sin(self.angles);
        rmat[:, 0, 1] = -rmat[:, 1, 0];

        stem0 = numpy.sum(rmat[:, numpy.newaxis, :, :]*stem0.reshape(N, 3, 1, 2), axis=3);
        stem1 = numpy.sum(rmat[:, numpy.newaxis, :, :]*stem1.reshape(N, 3, 1, 2), axis=3);
        tip = numpy.sum(rmat[:, numpy.newaxis, :, :]*tip.reshape(N, 3, 1, 2), axis=3);

        # put the triangles in the appropriate positions
        stem0 += self.positions[:, numpy.newaxis, :];
        stem1 += self.positions[:, numpy.newaxis, :];
        tip += self.positions[:, numpy.newaxis, :];

        colors = numpy.repeat(self.colors, 3, axis=0);
        vertices = numpy.concatenate([stem0, stem1, tip], axis=0);

        super(Arrows, self).__init__(vertices, colors=colors, color=color);


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
        self.polygon = polygon;
        self.outline = outline;

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
        #        tmp_poly = triangulate.triangulate(polygon, outline)
        # decompose the polygon into constituent triangles
        #        tmp_poly.calculate()
        # put the triangle vertices into a numpy array
        triangle_array = self.polygon.triangles
        textriangle_array = self.polygon.normalizedTriangles
        outline_array = self.outline.triangles
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
