import numpy
from freud.util import trimath

class outline:
    def __init__(self, vertsA, vertsB):
        self.vertsA = numpy.array(vertsA, dtype=numpy.float32);
        self.vertsB = numpy.array(vertsB, dtype=numpy.float32);
        if len(self.vertsA) < 3:
            raise TypeError("polygon A must have at least 3 vertices")
        if len(self.vertsB) < 3:
            raise TypeError("polygon B must have at least 3 vertices")
        if len(self.vertsA[1]) != 2:
            raise TypeError("positions for A must be an Nx2 array")
        if len(self.vertsB[1]) != 2:
            raise TypeError("positions for B must be an Nx2 array")
        if len(self.vertsA) != len(self.vertsB):
            raise TypeError("shape A and shape B must have the same number of vertices")
        self.n = len(self.vertsA)
        self.is_cclockwise(self.vertsA)
        self.is_cclockwise(self.vertsB)

    def is_cclockwise(self, verts):
        sign_sum = 0.0
        for j in range(self.n):
            # set up the two sets of verts
            if j == 0:
                i = self.n - 1
            else:
                i = j - 1
            if j == (self.n - 1):
                k = 0
            else:
                k = j + 1
            # math from stackoverflow
            # http://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
            ij = verts[i] - verts[j]
            jk = verts[k] - verts[j]
            edge_ij = numpy.array([ij[0], ij[1], 0], dtype=numpy.float32)
            edge_jk = numpy.array([jk[0], jk[1], 0], dtype=numpy.float32)
            tmp_sign = trimath.sinecheck(edge_ij, edge_jk)
            sign_sum += tmp_sign
        if not sign_sum < 0.0:
            raise TypeError("vertices must be listed in counter-clockwise order")

class polygon:
    # vertices must be in counter-clockwise order
    def __init__(self, verts):
        self.vertices = numpy.array(verts, dtype=numpy.float32);
        if len(self.vertices) < 3:
            raise TypeError("a polygon must have at least 3 vertices")
        if len(self.vertices[1]) != 2:
            raise TypeError("positions must be an Nx2 array")
        self.n = len(self.vertices)
        self.is_cclockwise()

    def is_cclockwise(self):
        sign_sum = 0.0
        for j in range(self.n):
            # set up the two sets of verts
            if j == 0:
                i = self.n - 1
            else:
                i = j - 1
            if j == (self.n - 1):
                k = 0
            else:
                k = j + 1
            # math from stackoverflow
            # http://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
            ij = self.vertices[i] - self.vertices[j]
            jk = self.vertices[k] - self.vertices[j]
            edge_ij = numpy.array([ij[0], ij[1], 0], dtype=numpy.float32)
            edge_jk = numpy.array([jk[0], jk[1], 0], dtype=numpy.float32)
            tmp_sign = trimath.sinecheck(edge_ij, edge_jk)
            sign_sum += tmp_sign
        if not sign_sum < 0.0:
            raise TypeError("vertices must be listed in counter-clockwise order")

    # This method is primarily used by the triangulation class in triangulate.py
    def remove_vert(self, i):
        #print("vertex removed:")
        #print(self.vertices[i])
        self.vertices = numpy.delete(self.vertices, (i), axis=0)
        self.n = len(self.vertices)

    def print_verts(self):
        for i in range(self.n):
            v = self.vertices[i]
            print("vertex {0}".format(i))
            print(v)

class triangle:

    def __init__(self, verts):
        self.vertices = numpy.array(verts, dtype=numpy.float32);
        if not len(self.vertices) == 3:
            raise TypeError("a triangle must have exactly 3 vertices")
        if not len(self.vertices[1]) == 2:
            raise TypeError("positions must be an 3x2 array")
        self.n = 3
        self.is_cclockwise()
    
    def is_cclockwise(self):
        sign_sum = 0.0
        for j in range(self.n):
            # set up the two sets of verts
            if j == 0:
                i = self.n - 1
            else:
                i = j - 1
            if j == (self.n - 1):
                k = 0
            else:
                k = j + 1
            # math from stackoverflow
            # http://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
            ij = self.vertices[i] - self.vertices[j]
            jk = self.vertices[k] - self.vertices[j]
            edge_ij = numpy.array([ij[0], ij[1], 0], dtype=numpy.float32)
            edge_jk = numpy.array([jk[0], jk[1], 0], dtype=numpy.float32)
            tmp_sign = trimath.sinecheck(edge_ij, edge_jk)
            sign_sum += tmp_sign
        if not sign_sum < 0.0:
            raise TypeError("vertices must be listed in counter-clockwise order")

    def print_verts(self):
        print("Triangle has the following vertices:")
        for i in range(self.n):
            v = self.vertices[i]
            print("vertex {0}".format(i))
            print(v)
