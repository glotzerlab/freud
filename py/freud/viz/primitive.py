from __future__ import division, print_function
import math
import numpy

from freud.viz import base

## \package freud.viz.primitive
#
# Definition of basic viz primitives
#

## Disk representation (2D)
#
# Represent N disks in 2D. Each has a given color and a global outline width is specified.
class Disks(base.Primitive):
    ## Initialize a disk representation
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

## Line representation
#
# Represent N lines in 2D or 3D (2D specific renderers may simply ignore the z component).
class Lines(base.Primitive):
    pass

## Repeated polygons
#
# Represent N instances of the same polygon in 2D, each at a different position, orientation, and color. Black edges
# are drawn given a global outline width.
#
class RepeatedPolygons(base.Primitive):
    ## Initialize a disk representation
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
        
        # -----------------------------------------------------------------
        # set up outline
        self.outline = outline;

