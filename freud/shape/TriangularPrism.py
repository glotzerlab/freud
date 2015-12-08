from __future__ import division
from numpy import sqrt
import numpy
from freud.shape import ConvexPolyhedron

# Example:
# from freud.shape.TriangularPrism import shape
points = [ 
          (-1/(2*sqrt(3)), -1/2, -1/2),
          (-1/(2*sqrt(3)), -1/2, 1/2),
          (-1/(2*sqrt(3)), 1/2, -1/2),
          (-1/(2*sqrt(3)), 1/2, 1/2),
          (1/sqrt(3), 0, -1/2),
          (1/sqrt(3), 0, 1/2),
         ]

shape = ConvexPolyhedron(numpy.array(points))
