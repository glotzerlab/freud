from __future__ import division
from numpy import sqrt
import numpy
from freud.shape import ConvexPolyhedron

# Example:
# from freud.shape.HexagonalPrism import shape
points = [ 
          (-1, 0, -1/2),
          (-1, 0, 1/2),
          (-1/2, -sqrt(3)/2, -1/2),
          (-1/2, -sqrt(3)/2, 1/2),
          (-1/2, sqrt(3)/2, -1/2),
          (-1/2, sqrt(3)/2, 1/2),
          (1/2, -sqrt(3)/2, -1/2),
          (1/2, -sqrt(3)/2, 1/2),
          (1/2, sqrt(3)/2, -1/2),
          (1/2, sqrt(3)/2, 1/2),
          (1, 0, -1/2),
          (1, 0, 1/2),
         ]

shape = ConvexPolyhedron(numpy.array(points))
