from collections import deque
from itertools import islice
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

def twiceTriangleArea(p0, p1, p2):
    """Returns twice the signed area of the triangle specified by the
    2D numpy points (p0, p1, p2)."""
    p1 = p1 - p0
    p2 = p2 - p0
    return p1[0]*p2[1] - p2[0]*p1[1]

def linesCross(line1, line2, tol=1e-5):
    """Returns true if two lines (specified as pairs of 2D numpy
    points) cross. Also takes a tolerance cutoff for [twice] the area
    between three colinear points"""
    # sign of triangle (line1[0], line2[0], line2[1])
    area1 = twiceTriangleArea(line1[0], line2[0], line2[1])
    sign1 = numpy.abs(area1) > tol and int(numpy.sign(area1))

    # sign of triangle (line1[1], line2[0], line2[1])
    area2 = twiceTriangleArea(line1[1], line2[0], line2[1])
    sign2 = numpy.abs(area2) > tol and int(numpy.sign(area2))

    # sign of triangle (line2[0], line1[0], line1[1])
    area3 = twiceTriangleArea(line2[0], line1[0], line1[1])
    sign3 = numpy.abs(area3) > tol and int(numpy.sign(area3))

    # sign of triangle (line2[1], line1[0], line1[1])
    area4 = twiceTriangleArea(line2[1], line1[0], line1[1])
    sign4 = numpy.abs(area4) > tol and int(numpy.sign(area4))

    # specify that if two lines share a vertex (one of the areas is
    # 0), they don't cross
    return all([sign1, sign2, sign3, sign4, (sign1 != sign2), (sign3 != sign4)])

def linesTouch(line1, line2):
    """Returns true if two lines (specified as pairs of 2D numpy
    points) touch, including sharing a vertex at two endpoints. Also
    takes a tolerance cutoff for [twice] the area between three
    colinear points"""
    # sign of triangle (line1[0], line2[0], line2[1])
    area1 = twiceTriangleArea(line1[0], line2[0], line2[1])
    sign1 = numpy.sign(area1)

    # sign of triangle (line1[1], line2[0], line2[1])
    area2 = twiceTriangleArea(line1[1], line2[0], line2[1])
    sign2 = numpy.sign(area2)

    # sign of triangle (line2[0], line1[0], line1[1])
    area3 = twiceTriangleArea(line2[0], line1[0], line1[1])
    sign3 = numpy.sign(area3)

    # sign of triangle (line2[1], line1[0], line1[1])
    area4 = twiceTriangleArea(line2[1], line1[0], line1[1])
    sign4 = numpy.sign(area4)

    return (sign1 != sign2) and (sign3 != sign4)

class Polygon:
    """Basic class to hold a set of points for a 2D polygon"""
    def __init__(self, verts):
        """Initialize a polygon with a counterclockwise list of 2D
        points and checks that they are ordered counter-clockwise"""
        self.vertices = numpy.array(verts, dtype=numpy.float32);

        if len(self.vertices) < 3:
            raise TypeError("a polygon must have at least 3 vertices")
        if len(self.vertices[1]) != 2:
            raise TypeError("positions must be an Nx2 array")
        self.n = len(self.vertices)

        # This actually checks that the majority of the polygon is
        # listed in counter-clockwise order, but seems like it should
        # be sufficient for common use cases. Non-simple polygons can
        # still sneak in clockwise vertices.
        if self.area() < 0:
            raise RuntimeError("Polygon was given with some clockwise vertices, "
                               "but it requires that vertices be listed in "
                               "counter-clockwise order")

    def area(self):
        """Calculate the signed area of the polygon with
        counterclockwise shapes having positive area"""
        shifted = numpy.roll(self.vertices, -1, axis=0)

        # areas is twice the signed area of each triangle in the shape
        areas = self.vertices[:, 0]*shifted[:, 1] - shifted[:, 0]*self.vertices[:, 1]

        return numpy.sum(areas)/2

    def center(self):
        """Center this polygon around (0, 0)"""
        self.vertices -= numpy.mean(self.vertices, axis=0)

    def print_verts(self):
        for vert in self.vertices:
            print("vertex {}".format(vert))

    def splitMonotone(self):
        """Returns a list of subpolygons of this polygon where each
        subpolygon is guaranteed to be monotone with respect to any
        vertical line."""
        sortIdx = numpy.argsort(self.vertices[:, 0], kind='mergesort')
        print(numpy.diff(sortIdx))
        print(numpy.abs(numpy.diff(sortIdx)) == 1)
        print(self.vertices[sortIdx])

    @property
    def triangles(self):
        """A cached property of an Ntx3x2 numpy array of points, where
        Nt is the number of triangles in this polygon."""
        try:
            return self._triangles
        except AttributeError:
            self._triangles = self._triangulate()
        return self._triangles

    def _triangulate(self):
        """Return a list of triangles for the 3 2D points of Nt
        triangles."""

        if self.n <= 3:
            return [tuple(self.vertices)]

        result = []
        remaining = self.vertices

        # step around the shape and grab ears until only 4 vertices are left
        while len(remaining) > 4:
            signs = []
            for vert in (remaining[-1], remaining[1]):
                arms1 = remaining[2:-2] - vert
                arms2 = vert - remaining[3:-1]
                signs.append(numpy.sign(arms1[:, 1]*arms2[:, 0] - arms2[:, 1]*arms1[:, 0]))
            for rest in (remaining[2:-2], remaining[3:-1]):
                arms1 = remaining[-1] - rest
                arms2 = rest - remaining[1]
                signs.append(numpy.sign(arms1[:, 1]*arms2[:, 0] - arms2[:, 1]*arms1[:, 0]))

            cross = numpy.any(numpy.bitwise_and(signs[0] != signs[1], signs[2] != signs[3]))
            if not cross and twiceTriangleArea(remaining[-1], remaining[0], remaining[1]) > 0.:
                # triangle [-1, 0, 1] is a good one, cut it out
                result.append((remaining[-1].copy(), remaining[0].copy(), remaining[1].copy()))
                remaining = remaining[1:]
            else:
                remaining = numpy.roll(remaining, 1, axis=0)

        # there must now be 0 or 1 concave vertices left; find the
        # concave vertex (or a vertex) and fan out from it
        vertices = remaining
        shiftedUp = vertices - numpy.roll(vertices, 1, axis=0)
        shiftedBack = numpy.roll(vertices, -1, axis=0) - vertices

        # signed area for each triangle (i-1, i, i+1) for vertex i
        areas = shiftedBack[:, 1]*shiftedUp[:, 0] - shiftedUp[:, 1]*shiftedBack[:, 0]

        concave = numpy.where(areas < 0.)[0]

        fan = (concave[0] if len(concave) else 0)
        fanVert = remaining[fan]
        remaining = numpy.roll(remaining, -fan, axis=0)[1:]

        result.extend([(fanVert, remaining[0], remaining[1]),
                       (fanVert, remaining[1], remaining[2])])

        return numpy.array(result)

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
