from __future__ import division
from numpy import sqrt
import numpy
from freud.shape import ConvexPolyhedron

# Example:
# from freud.shape.Octahedron.py import shape
points = [ 
          (-(1/sqrt(2)), 0, 0),
          (0, 1/sqrt(2), 0),
          (0, 0, -(1/sqrt(2))),
          (0, 0, 1/sqrt(2)),
          (0, -(1/sqrt(2)), 0),
          (1/sqrt(2), 0, 0),
         ]

shape = ConvexPolyhedron(numpy.array(points))
