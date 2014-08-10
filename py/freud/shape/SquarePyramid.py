from __future__ import division
from numpy import sqrt
import numpy
from freud.shape import ConvexPolyhedron

# Example:
# from freud.shape.SquarePyramid import shape
points = [ 
          (0, 0, 1/sqrt(2)),
          (0, -(1/sqrt(2)), 0),
          (0, 1/sqrt(2), 0),
          (-(1/sqrt(2)), 0, 0),
          (1/sqrt(2), 0, 0),
         ]

# The origin is on the surface of the above hull
# We don't necessarily need the origin to be the centroid, but it is convenient for
# the origin to be an interior point.
points = numpy.array(points)
points -= [0, 0, 1/(2*sqrt(2))]

shape = ConvexPolyhedron(numpy.array(points))
