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

from freud.shape import Polygon

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
        self.updated = [];
        self.update(positions=positions, diameters=diameters, colors=colors, color=color, outline=outline);

        # -----------------------------------------------------------------
        # set up outline
        self.outline = outline;


    def update(self, positions=None, diameters=None, colors=None, color=None, outline=0.1):
        updated = set(self.updated);

        # -----------------------------------------------------------------
        # set up positions
        # convert to a numpy array
        if positions is not None:
            self.positions = numpy.array(positions, dtype=numpy.float32);
            # error check the input
            if len(self.positions.shape) != 2:
                raise TypeError("positions must be a Nx2 array");
            if self.positions.shape[1] != 2:
                raise ValueError("positions must be a Nx2 array");

            self.N = self.positions.shape[0];
            updated.add('position');

        # -----------------------------------------------------------------
        # set up diameters
        try:
            self.diameters;
        except AttributeError:
            self.diameters = numpy.zeros(shape=(self.N,), dtype=numpy.float32);
            self.diameters[:] = 1;
            updated.add('diameter');
        if diameters is not None:
            self.diameters = numpy.array(diameters);

            # error check diameters
            if len(self.diameters.shape) != 1:
                raise TypeError("diameters must be a single dimension array");
            if self.diameters.shape[0] != self.N:
                raise ValueError("diameters must have N the same as positions");
            updated.add('diameter');

        # -----------------------------------------------------------------
        # set up colors
        try:
            self.colors
        except AttributeError:
            self.colors = numpy.zeros(shape=(self.N,4), dtype=numpy.float32);
            self.colors[:,3] = 1;
            updated.add('color');
        if colors is not None:
            self.colors = numpy.array(colors, dtype=numpy.float32);

            # error check colors
            if len(self.colors.shape) != 2:
                raise TypeError("colors must be a Nx4 array");
            if self.colors.shape[1] != 4:
                raise ValueError("colors must have N the same as positions");
            if self.colors.shape[0] != self.N:
                raise ValueError("colors must have N the same as positions");
            updated.add('color');

        if color is not None:
            acolor = numpy.array(color);
            if len(acolor.shape) != 1:
                raise TypeError("color must be a 4 element array");
            if acolor.shape[0] != 4:
                raise ValueError("color must be a 4 element array");

            self.colors[:,:] = acolor;
            updated.add('color');

        self.updated = list(updated);


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
        self.updated = [];
        self.update(vertices=vertices, texcoords=texcoords, colors=colors,
                    color=color, tex_fname=tex_fname);

    def update(self, vertices=None, texcoords=None, colors=None, color=None,
               tex_fname=None):
        updated = set(self.updated);

        # -----------------------------------------------------------------
        # set up vertices
        # convert to a numpy array
        if vertices is not None:
            self.vertices = numpy.array(vertices, dtype=numpy.float32);
            # error check the input
            if len(self.vertices.shape) != 3:
                raise TypeError("vertices must be a Nx3x2 array");
            if self.vertices.shape[1] != 3:
                raise ValueError("vertices must be a Nx3x2 array");
            if self.vertices.shape[2] != 2:
                raise ValueError("vertices must be a Nx3x2 array");

            self.N = self.vertices.shape[0];
            updated.add('position');

        # -----------------------------------------------------------------
        # set up texcoords
        if texcoords is not None:
            self.texcoords = numpy.array(texcoords, dtype=numpy.float32);

            if len(self.texcoords.shape) != 3:
                raise TypeError("texcoords must be a Nx3x2 array");
            if self.texcoords.shape[1] != 3:
                raise ValueError("texcoords must be a Nx3x2 array");
            if self.texcoords.shape[2] != 2:
                raise ValueError("texcoords must be a Nx3x2 array");

            updated.add('texcoord');
        try:
            self.texcoords;
        except AttributeError:
            self.texcoords = numpy.zeros(shape=self.vertices.shape, dtype=numpy.float32);
            updated.add('texcoord');

        self.tex_fname = tex_fname;

        # -----------------------------------------------------------------
        # set up colors
        if colors is not None:
            self.colors = numpy.array(colors, dtype=numpy.float32);

            # error check colors
            if len(self.colors.shape) != 2:
                raise TypeError("colors must be a Nx4 array");
            if self.colors.shape[1] != 4:
                raise ValueError("colors must be a Nx4 array");
            if self.colors.shape[0] != self.N:
                raise ValueError("colors must have N the same as positions");
            updated.add('color');

        try:
            self.colors;
        except AttributeError:
            self.colors = numpy.zeros(shape=(self.N,4), dtype=numpy.float32);
            self.colors[:,3] = 1;
            updated.add('color');

        if color is not None:
            acolor = numpy.array(color);
            if len(acolor.shape) != 1:
                raise TypeError("color must be a 4 element array");
            if acolor.shape[0] != 4:
                raise ValueError("color must be a 4 element array");

            self.colors[:,:] = acolor;
            updated.add('color');

        self.updated = list(updated);


## Line primitive
#
# Represent N lines in 2D. This primitive draws N line segments (with
# square ends) given a set of start and end points.
class Lines(Triangles):
    ## Initialize a line primitive
    # \param starts Nx2 array listing the origin of each line segment (in distance units)
    # \param width line width to draw (in distance units)
    # \param ends Either an Nx2 array listing the destination of each line segment (in distance units) or None to connect all points in a single continuous line
    # \param colors Nx4 array listing the colors (rgba 0.0-1.0) of each vertex (in SRGB)
    # \param color 4 element iterable listing the color to be applied to every vertex (in SRGB)
    #              \a color overrides anything set by colors
    # \param maxLength If any line is longer than maxLength, don't draw it
    #
    # When colors is None, it defaults to (0,0,0,1) for each particle.
    #
    # \note N **must** be the same for each array
    #
    # After initialization, the instance will have members starts,
    # ends, and colors, each being a numpy array of the appropriate
    # size and dtype float32. Users should not modify these directly,
    # they are intended for use only by renderers. Instead, users
    # should create a new primitive from scratch to rebuild geometry.
    def __init__(self, starts, width=1.0, ends=None, colors=None, color=None, maxLength=None):
        base.Primitive.__init__(self);
        self.singleLine = (ends is None);
        self.maxLength = maxLength;
        self.updated = [];
        self.update(starts=starts, width=width, ends=ends, colors=colors,
                    color=color);

    def update(self, starts=None, width=None, ends=None, colors=None, color=None):
        updated = set(self.updated);

        # -----------------------------------------------------------------
        # set up starts
        # convert to a numpy array
        if starts is not None:
            if self.singleLine:
                self.starts = numpy.array(starts[:-1], dtype=numpy.float32);
                self.ends = numpy.array(starts[1:], dtype=numpy.float32);

                # error check the input
                if len(self.starts.shape) != 2:
                    raise TypeError("starts must be a Nx2 array");
                if self.starts.shape[1] != 2:
                    raise ValueError("starts must be a Nx2 array");
                if self.starts.shape[0] < 1:
                    raise ValueError("For a continuous line, starts must be of "
                                     "length 2 or greater");

                self.N = self.starts.shape[0];
                updated.add('position');
            else:
                self.starts = numpy.array(starts, dtype=numpy.float32);
                # error check the input
                if len(self.starts.shape) != 2:
                    raise TypeError("starts must be a Nx2 array");
                if self.starts.shape[1] != 2:
                    raise ValueError("starts must be a Nx2 array");

                self.N = self.starts.shape[0];
                updated.add('position');

        # -----------------------------------------------------------------
        # set up width
        if width is not None:
            self.width = numpy.array(width, dtype=numpy.float32);

            # error check width
            if len(self.width.shape) != 0:
                raise TypeError("width must be a scalar");

            updated.add('position');

        # -----------------------------------------------------------------
        # set up ends
        # convert to a numpy array
        if ends is not None and not self.singleLine:
            self.ends = numpy.array(ends, dtype=numpy.float32);
            # error check the input
            if len(self.ends.shape) != 2:
                raise TypeError("ends must be a Nx2 array");
            if self.ends.shape[0] != self.N or self.ends.shape[1] != 2:
                raise ValueError("ends must be a Nx2 array");

            updated.add('position');

        # -----------------------------------------------------------------
        # set up colors
        try:
            self.arrColors;
        except AttributeError:
            self.arrColors = numpy.zeros(shape=(self.N,4), dtype=numpy.float32);
            self.arrColors[:,3] = 1;

        if colors is not None:
            self.arrColors = numpy.array(colors, dtype=numpy.float32);

            # Silently fix input size in single-line mode
            if self.singleLine and self.arrColors.shape[0] == self.N + 1:
                self.arrColors = self.arrColors[:-1];

            # error check colors
            if len(self.arrColors.shape) != 2:
                raise TypeError("colors must be a Nx4 array");
            if self.arrColors.shape[1] != 4:
                raise ValueError("colors must be a Nx4 array");
            if self.arrColors.shape[0] != self.N:
                raise ValueError("colors must have N the same as positions");

            updated.add('color');

        if color is not None:
            acolor = numpy.array(color);
            if len(acolor.shape) != 1:
                raise TypeError("color must be a 4 element array");
            if acolor.shape[0] != 4:
                raise ValueError("color must be a 4 element array");

            updated.add('color');

        if 'position' in updated:
            # stem0 and stem1 are the two triangles for the rectangular
            # "stem" of the line
            stem0 = numpy.array([[[0, -.5*self.width],
                                  [0, .5*self.width],
                                  [1, .5*self.width]]],
                                dtype=numpy.float32);
            stem1 = numpy.array([[[0, -.5*self.width],
                                  [1, .5*self.width],
                                  [1, -.5*self.width]]],
                                dtype=numpy.float32);

            deltas = self.ends - self.starts;
            self.lengths = numpy.sqrt(numpy.sum(deltas**2, axis=1)) + .5*self.width;
            self.angles = numpy.arctan2(deltas[:, 1], deltas[:, 0]);

            # "delete" segments that are too long (set length to 0)
            if self.maxLength is not None:
                self.lengths[self.lengths > self.maxLength] = 0.;

            # replicate the "base image" into the proper shape
            stem0 = numpy.repeat(stem0, self.N, axis=0);
            stem1 = numpy.repeat(stem1, self.N, axis=0);

            # scale the length of stem by the given line lengths
            stem0[:, 2, 0] *= self.lengths;
            stem1[:, 1:, 0] *= self.lengths[:, numpy.newaxis];

            # rotate the vertices into the correct orientation
            rmat = numpy.empty((self.N, 2, 2));
            rmat[:, 0, 0] = rmat[:, 1, 1] = numpy.cos(self.angles);
            rmat[:, 1, 0] = numpy.sin(self.angles);
            rmat[:, 0, 1] = -rmat[:, 1, 0];

            stem0 = numpy.sum(rmat[:, numpy.newaxis, :, :]*stem0.reshape(self.N, 3, 1, 2), axis=3);
            stem1 = numpy.sum(rmat[:, numpy.newaxis, :, :]*stem1.reshape(self.N, 3, 1, 2), axis=3);

            # put the triangles in the appropriate positions
            stem0 += self.starts[:, numpy.newaxis, :];
            stem1 += self.starts[:, numpy.newaxis, :];

            vertices = numpy.concatenate([stem0, stem1], axis=0);

        if 'color' in updated:
            colors = numpy.repeat(self.arrColors, 3, axis=0);

        super(Lines, self).update(vertices=vertices, colors=colors, color=color);


## Rotated Triangle primitive
#
# Represent N shapes in 2D specified by positions, orientations, a set
# of triangles with local vertex images, and a color.
#
class Polygons(base.Primitive):
    ## Initialize a rotated triangle primitive
    # \param polygon Polygon object containing the local vertices of a shape
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
    def __init__(self, polygon, positions, orientations, texcoords=None, colors=None, color=None, tex_fname=None):
        base.Primitive.__init__(self);
        self.updated = [];
        self.update(polygon, positions, orientations, texcoords=texcoords, colors=colors, color=color, tex_fname=tex_fname);

    def update(self, polygon=None, positions=None, orientations=None, texcoords=None, colors=None, color=None, tex_fname=None):
        updated = set(self.updated);

        # -----------------------------------------------------------------
        # set up polygon image
        # convert to a numpy array
        if polygon is not None:
            self.image = numpy.array(polygon.triangles, dtype=numpy.float32);
            # error check the input
            if len(self.image.shape) != 3:
                raise TypeError("image must be a Ntx3x2 array; Polygon's "
                                "triangles must have been corrupted!");
            if self.image.shape[1] != 3:
                raise ValueError("image must be a Ntx3x2 array; Polygon's "
                                 "triangles must have been corrupted!");
            if self.image.shape[2] != 2:
                raise ValueError("image must be a Ntx3x2 array; Polygon's "
                                 "triangles must have been corrupted!");

            Nt = self.image.shape[0];
            updated.add('images');

            try:
                self.images = numpy.tile(self.image, (self.Np, 1, 1)).reshape((3*self.Np*Nt, 2));
                if Nt != self.Nt:
                    raise RuntimeError('Polygons.update() does not '
                                       'support changing the number of shapes');
            except AttributeError:
                # we're actually inside the constructor since self.Nt
                # doesn't exist yet; we will re-set self.images in a
                # few lines, after we set self.Np
                self.Nt = Nt;

        # -----------------------------------------------------------------
        # set up positions
        # convert to a numpy array
        if positions is not None:
            self.positions = numpy.array(positions, dtype=numpy.float32);

            # error check the input
            if len(self.positions.shape) != 2:
                raise TypeError("positions must be a Npx2 array");
            if self.positions.shape[1] != 2:
                raise ValueError("positions must be a Npx2 array");

            Np = self.positions.shape[0];
            updated.add('positions');
            self.positions = numpy.tile(
                self.positions[:, numpy.newaxis, :], (1, 3*self.Nt, 1)).reshape((3*Np*self.Nt, 2));

            try:
                if Np != self.Np:
                    raise RuntimeError('Polygons.update() does not '
                                       'support changing the number of shapes');
            except AttributeError:
                # we're actually inside the constructor since self.Np
                # doesn't exist yet
                self.Np = Np;
                self.images = numpy.tile(self.image, (self.Np, 1, 1)).reshape((3*self.Np*self.Nt, 2));
                self.colors = numpy.zeros((3*self.Np*self.Nt, 4), dtype=numpy.float32);
                self.colors[:, 3] = 1.0;
                updated.add('images');
                updated.add('colors');

        # -----------------------------------------------------------------
        # set up orientations
        # convert to a numpy array
        if orientations is not None:
            self.orientations = numpy.array(orientations, dtype=numpy.float32);

            # error check the input
            if len(self.orientations.shape) != 1:
                raise TypeError("orientations must be a Np-length array");
            if len(self.orientations) != self.Np:
                raise ValueError("Must have the same number of orientations as positions");

            updated.add('orientations');
            self.orientations = numpy.tile(
                self.orientations[:, numpy.newaxis], (1, 3*self.Nt)).reshape((3*self.Np*self.Nt, 1));

        if texcoords is not None:
            self.texcoords = numpy.array(texcoords, dtype=numpy.float32);

            if len(self.texcoords.shape) != 3:
                raise TypeError("texcoords must be a Nx3x2 array");
            if self.texcoords.shape[1] != 3:
                raise ValueError("texcoords must be a Nx3x2 array");
            if self.texcoords.shape[2] != 2:
                raise ValueError("texcoords must be a Nx3x2 array");

            updated.add('texcoords');

        try:
            self.texcoords;
        except AttributeError:
            self.texcoords = numpy.zeros(shape=(self.Np*self.Nt, 3, 2), dtype=numpy.float32);
            self.tex_fname = tex_fname;
            updated.add('texcoords');

        # -----------------------------------------------------------------
        # set up colors
        if colors is not None:
            # error check colors
            if len(colors.shape) != 2:
                raise TypeError("colors must be a Npx4 array");
            if colors.shape[1] != 4:
                raise ValueError("colors must be a Npx4 array");
            if colors.shape[0] != self.Np:
                raise ValueError("colors must have N the same as positions");

            colors = numpy.asarray(colors, dtype=numpy.float32);
            self.colors = numpy.tile(colors[:, numpy.newaxis, :],
                                     (1, 3*self.Nt, 1)).reshape((3*self.Np*self.Nt, 4));
            updated.add('colors');

        if color is not None:
            acolor = numpy.array(color, dtype=numpy.float32);
            if len(acolor.shape) != 1:
                raise TypeError("color must be a 4 element array");
            if acolor.shape[0] != 4:
                raise ValueError("color must be a 4 element array");

            self.colors = numpy.tile(acolor[numpy.newaxis, :], (3*self.Np*self.Nt, 1));

            updated.add('colors');

        self.updated = list(updated);


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
        base.Primitive.__init__(self);
        self.updated = [];
        self.update(positions=positions, widths=widths, lengths=lengths,
                    angles=angles, colors=colors, color=color, aspectRatio=aspectRatio);


    def update(self, positions=None, widths=None, lengths=None, angles=None,
               colors=None, color=None, aspectRatio=5.):
        updated = set(self.updated);

        # -----------------------------------------------------------------
        # set up positions
        # convert to a numpy array
        if positions is not None:
            self.positions = numpy.array(positions, dtype=numpy.float32);
            # error check the input
            if len(self.positions.shape) != 2:
                raise TypeError("positions must be a Nx2 array");
            if self.positions.shape[1] != 2:
                raise ValueError("positions must be a Nx2 array");

            self.N = self.positions.shape[0];
            updated.add('position');

        # -----------------------------------------------------------------
        # set up widths
        if widths is not None:
            self.widths = numpy.array(widths, dtype=numpy.float32);
            if len(self.widths.shape) == 0:
                self.widths = numpy.repeat(self.widths, self.N);

            # error check widths
            if len(self.widths.shape) != 1:
                raise TypeError("widths must be a scalar or single dimension array");
            if self.widths.shape[0] != self.N:
                raise ValueError("widths must have N the same as positions or 1");

            updated.add('position');

        # -----------------------------------------------------------------
        # set up lengths
        if lengths is not None:
            self.lengths = numpy.array(lengths, dtype=numpy.float32);
            if len(self.lengths.shape) == 0:
                self.lengths = numpy.repeat(self.lengths, self.N);

            # error check lengths
            if len(self.lengths.shape) != 1:
                raise TypeError("lengths must be a scalar or single dimension array");
            if self.lengths.shape[0] != self.N:
                raise ValueError("lengths must have N the same as positions or 1");

            updated.add('position');

        # -----------------------------------------------------------------
        # set up angles
        if angles is not None:
            self.angles = numpy.array(angles, dtype=numpy.float32);

            # error check angles
            if len(self.angles.shape) not in [0, 1]:
                raise TypeError("angles must be a scalar or single dimension array");
            if len(self.angles.shape) and self.angles.shape[0] != self.N:
                raise ValueError("angles must have N the same as positions or 1");

            updated.add('position');

        # -----------------------------------------------------------------
        # set up colors
        try:
            self.arrColors;
        except AttributeError:
            self.arrColors = numpy.zeros(shape=(self.N,4), dtype=numpy.float32);
            self.arrColors[:,3] = 1;

        if colors is not None:
            self.arrColors = numpy.array(colors, dtype=numpy.float32);

            # error check colors
            if len(self.arrColors.shape) != 2:
                raise TypeError("colors must be a Nx4 array");
            if self.arrColors.shape[1] != 4:
                raise ValueError("colors must be a Nx4 array");
            if self.arrColors.shape[0] != self.N:
                raise ValueError("colors must have N the same as positions");

            updated.add('color');

        if color is not None:
            acolor = numpy.array(color);
            if len(acolor.shape) != 1:
                raise TypeError("color must be a 4 element array");
            if acolor.shape[0] != 4:
                raise ValueError("color must be a 4 element array");

            updated.add('color');


        if 'position' in updated:
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

            stem0 = numpy.repeat(stem0, self.N, axis=0);
            stem1 = numpy.repeat(stem1, self.N, axis=0);
            tip = numpy.repeat(tip, self.N, axis=0);

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
            rmat = numpy.empty((self.N, 2, 2));
            rmat[:, 0, 0] = rmat[:, 1, 1] = numpy.cos(self.angles);
            rmat[:, 1, 0] = numpy.sin(self.angles);
            rmat[:, 0, 1] = -rmat[:, 1, 0];

            stem0 = numpy.sum(rmat[:, numpy.newaxis, :, :]*stem0.reshape(self.N, 3, 1, 2), axis=3);
            stem1 = numpy.sum(rmat[:, numpy.newaxis, :, :]*stem1.reshape(self.N, 3, 1, 2), axis=3);
            tip = numpy.sum(rmat[:, numpy.newaxis, :, :]*tip.reshape(self.N, 3, 1, 2), axis=3);

            # put the triangles in the appropriate positions
            stem0 += self.positions[:, numpy.newaxis, :];
            stem1 += self.positions[:, numpy.newaxis, :];
            tip += self.positions[:, numpy.newaxis, :];

            vertices = numpy.concatenate([stem0, stem1, tip], axis=0);

        if 'color' in updated:
            colors = numpy.repeat(self.arrColors, 3, axis=0);

        super(Arrows, self).update(vertices=vertices, colors=colors, color=color);


## Repeated polygons shim for compatibility
#
# Represent N instances of the same polygon in 2D, each at a different position, orientation, and color.
#
class RepeatedPolygons(Polygons):
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
    def __init__(self, positions, angles, polygon, colors=None, color=None,
                 outline=0.1, tex_fname=None):
        self.polygon = Polygon(polygon);
        super(RepeatedPolygons, self).__init__(
            positions=positions, orientations=angles, polygon=self.polygon,
            colors=colors, color=color, tex_fname=tex_fname);


## Spheropolygons shim for compatibility
class Spheropolygons(RepeatedPolygons):
    def __init__(self, positions, angles, polygon, colors=None, color=None,
                 outline=0.1, radius=1., tex_fname=None):
        self.polygon = Polygon(polygon).getRounded(radius);
        super(Spheropolygons, self).__init__(
            positions, angles, self.polygon.vertices, colors=colors,
            color=color, outline=outline, tex_fname=tex_fname);


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
        self.updated = []

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
