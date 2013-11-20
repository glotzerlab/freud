from __future__ import division
from numpy import sqrt
import numpy
from freud.shape import ConvexPolyhedron

# Example:
# from freud.shape.TruncatedCube import shape
points = [ 
          (-1/2, 1/2 + 1/sqrt(2), 1/2 + 1/sqrt(2)),
          (-1/2, 1/2 + 1/sqrt(2), (2 - 2*sqrt(2))**(-1)),
          (-1/2, (2 - 2*sqrt(2))**(-1), 1/2 + 1/sqrt(2)),
          (-1/2, (2 - 2*sqrt(2))**(-1), (2 - 2*sqrt(2))**(-1)),
          (1/2, 1/2 + 1/sqrt(2), 1/2 + 1/sqrt(2)),
          (1/2, 1/2 + 1/sqrt(2), (2 - 2*sqrt(2))**(-1)),
          (1/2, (2 - 2*sqrt(2))**(-1), 1/2 + 1/sqrt(2)),
          (1/2, (2 - 2*sqrt(2))**(-1), (2 - 2*sqrt(2))**(-1)),
          (1/2 + 1/sqrt(2), -1/2, 1/2 + 1/sqrt(2)),
          (1/2 + 1/sqrt(2), -1/2, (2 - 2*sqrt(2))**(-1)),
          (1/2 + 1/sqrt(2), 1/2, 1/2 + 1/sqrt(2)),
          (1/2 + 1/sqrt(2), 1/2, (2 - 2*sqrt(2))**(-1)),
          (1/2 + 1/sqrt(2), 1/2 + 1/sqrt(2), -1/2),
          (1/2 + 1/sqrt(2), 1/2 + 1/sqrt(2), 1/2),
          (1/2 + 1/sqrt(2), (2 - 2*sqrt(2))**(-1), -1/2),
          (1/2 + 1/sqrt(2), (2 - 2*sqrt(2))**(-1), 1/2),
          ((2 - 2*sqrt(2))**(-1), -1/2, 1/2 + 1/sqrt(2)),
          ((2 - 2*sqrt(2))**(-1), -1/2, (2 - 2*sqrt(2))**(-1)),
          ((2 - 2*sqrt(2))**(-1), 1/2, 1/2 + 1/sqrt(2)),
          ((2 - 2*sqrt(2))**(-1), 1/2, (2 - 2*sqrt(2))**(-1)),
          ((2 - 2*sqrt(2))**(-1), 1/2 + 1/sqrt(2), -1/2),
          ((2 - 2*sqrt(2))**(-1), 1/2 + 1/sqrt(2), 1/2),
          ((2 - 2*sqrt(2))**(-1), (2 - 2*sqrt(2))**(-1), -1/2),
          ((2 - 2*sqrt(2))**(-1), (2 - 2*sqrt(2))**(-1), 1/2),
         ]

shape = ConvexPolyhedron(numpy.array(points))
