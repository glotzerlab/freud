from __future__ import division
from numpy import sqrt
import numpy
from freud.shape import ConvexPolyhedron

# Example:
# from freud.shape.Dodecahedron import shape

phi = (1. + sqrt(5.))/2.
inv = 2./(1. + sqrt(5.))
points = [
          (-1,-1,-1),
          (-1,-1, 1),
          (-1, 1,-1),
          (-1, 1, 1),
          ( 1,-1,-1),
          ( 1,-1, 1),
          ( 1, 1,-1),
          ( 1, 1, 1),
          ( 0,-inv,-phi),
          ( 0,-inv, phi),
          ( 0, inv,-phi),
          ( 0, inv, phi),
          (-inv,-phi, 0),
          (-inv, phi, 0),
          ( inv,-phi, 0),
          ( inv, phi, 0),
          (-phi, 0,-inv),
          (-phi, 0, inv),
          ( phi, 0,-inv),
          ( phi, 0, inv)
         ]
# produces a dodecahedron with circumradius sqrt(3)

shape = ConvexPolyhedron(numpy.array(points))

