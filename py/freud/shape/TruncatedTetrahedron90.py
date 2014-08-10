from __future__ import division
from numpy import sqrt
import numpy
from freud.shape import ConvexPolyhedron

# Example:
# from freud.shape.TruncatedTetrahedron90 import shape
points = [ 
         (1, 0.1, 0.1), 
         (0.1, 1, 0.1), 
         (0.1, 0.1, 1), 
         (1, -0.1, -0.1), 
         (0.1, -0.1, -1), 
         (0.1, -1, -0.1), 
         (-0.1, 1, -0.1), 
         (-0.1, 0.1, -1), 
         (-1, 0.1, -0.1), 
         (-0.1, -0.1, 1), 
         (-0.1, -1, 0.1), 
         (-1, -0.1, 0.1)
         ]

shape = ConvexPolyhedron(numpy.array(points))
