from __future__ import division
from numpy import sqrt
import numpy
from freud.shape import ConvexPolyhedron

# Example:
# from freud.shape.Cube import shape

points = [ (1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),
           (1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1) ]

shape = ConvexPolyhedron(numpy.array(points))

