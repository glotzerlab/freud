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

    def getRounded(self, radius=1.0, granularity=5):
        """Approximate a spheropolygon by adding rounding to the
        corners. Returns a new Polygon object."""
        # Make 3D unit vectors drs from each vertex i to its neighbor i+1
        drs = numpy.roll(self.vertices, -1, axis=0) - self.vertices;
        drs /= numpy.sqrt(numpy.sum(drs*drs, axis=1))[:, numpy.newaxis];
        drs = numpy.hstack([drs, numpy.zeros((drs.shape[0], 1))]);

        # relStarts and relEnds are the offsets relative to the first and
        # second point of each line segment in the polygon.
        rvec = numpy.array([[0, 0, -1]])*radius;
        relStarts = numpy.cross(rvec, drs)[:, :2];
        relEnds =  numpy.cross(rvec, drs)[:, :2];

        # absStarts and absEnds are the beginning and end points for each
        # straight line segment.
        absStarts = self.vertices + relStarts;
        absEnds = numpy.roll(self.vertices, -1, axis=0) + relEnds;

        relStarts = numpy.roll(relStarts, -1, axis=0);

        # We will join each of these segments by a round cap; this will be
        # done by tracing an arc with the given radius, centered at each
        # vertex from an end of a line segment to a beginning of the next
        theta1s = numpy.arctan2(relEnds[:, 1], relEnds[:, 0]);
        theta2s = numpy.arctan2(relStarts[:, 1], relStarts[:, 0]);
        dthetas = (theta2s - theta1s) % (2*numpy.pi);

        # thetas are the angles at which we'll place points for each
        # vertex; curves are the points on the approximate curves on the
        # corners.
        thetas = numpy.zeros((self.vertices.shape[0], granularity));
        for i, (theta1, dtheta) in enumerate(zip(theta1s, dthetas)):
            thetas[i] = theta1 + numpy.linspace(0, dtheta, 2 + granularity)[1:-1];
        curves = radius*numpy.vstack([numpy.cos(thetas).flat, numpy.sin(thetas).flat]).T;
        curves = curves.reshape((-1, granularity, 2));
        curves += numpy.roll(self.vertices, -1, axis=0)[:, numpy.newaxis, :];

        # Now interleave the pieces
        result = [];
        for (end, curve, start, vert, dtheta) in zip(absEnds, curves,
                                                     numpy.roll(absStarts, -1, axis=0),
                                                     numpy.roll(self.vertices, -1, axis=0),
                                                     dthetas):
            # convex case: add the end of the last straight line
            # segment, the curved edge, then the start of the next
            # straight line segment.
            if dtheta <= numpy.pi:
                result.append(end);
                result.append(curve);
                result.append(start);
            # concave case: don't use the curved region, just find the
            # intersection and add that point.
            else:
                l = radius/numpy.cos(dtheta/2);
                p = 2*vert - start - end;
                p /= trimath.norm(p);
                result.append(vert + p*l);

        result = numpy.vstack(result);

        return Polygon(result);

    def print_verts(self):
        for vert in self.vertices:
            print("vertex {}".format(vert))

    @property
    def triangles(self):
        """A cached property of an Ntx3x2 numpy array of points, where
        Nt is the number of triangles in this polygon."""
        try:
            return self._triangles
        except AttributeError:
            self._triangles = self._triangulation()
        return self._triangles

    @property
    def normalizedTriangles(self):
        try:
            return self._normalizedTriangles
        except AttributeError:
            self._normalizedTriangles = self._triangles.copy()
            self._normalizedTriangles -= numpy.min(self._triangles)
            self._normalizedTriangles /= numpy.max(self._normalizedTriangles)
        return self._normalizedTriangles

    def _triangulation(self):
        """Return a numpy array of triangles with shape (Nt, 3, 2) for
        the 3 2D points of Nt triangles."""

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

        return numpy.array(result, dtype=numpy.float32)

class Outline(object):
    def __init__(self, polygon, width):
        """Initialize an outline of a given Polygon object. Takes the
        polygon in question and the outline width to inset."""
        self.polygon = polygon
        self.width = width

    @property
    def width(self):
        """Property for the width of the outline. Updates the
        triangulation when set."""
        return self._width

    @width.setter
    def width(self, width):
        self._width = width
        self._triangulate()

    def _triangulate(self):
        """Triangulates an Outline object. Sets the triangles field to
        a Ntx3x2 numpy array of triangles."""
        drs = numpy.roll(self.polygon.vertices, -1, axis=0) - self.polygon.vertices
        ns = drs/numpy.sqrt(numpy.sum(drs*drs, axis=1)).reshape((len(drs), 1))
        thetas = numpy.arctan2(drs[:, 1], drs[:, 0])
        dthetas = (thetas - numpy.roll(thetas, 1))%(2*numpy.pi)

        concave = dthetas > numpy.pi
        convex = concave == False

        hs = numpy.repeat(self.width, len(drs))
        hs[convex] /= numpy.cos(dthetas[convex]/2)
        # flip the concave bisectors
        hs[concave] *= -1
        hs[concave] /= numpy.sin(dthetas[concave]/2)
        hs = hs.reshape((len(hs), 1))

        bisectors = ns - numpy.roll(ns, 1, axis=0)
        bisectors /= numpy.sqrt(numpy.sum(bisectors*bisectors, axis=1)).reshape((len(ns), 1))

        inners = self.polygon.vertices + hs.reshape((len(ns), 1))*bisectors

        result = numpy.empty((2, self.polygon.n, 3, 2), dtype=numpy.float32)
        result[:, :, 0, :] = (self.polygon.vertices, inners)
        result[:, :, 1, :] = numpy.roll(self.polygon.vertices, -1, axis=0)
        result[:, :, 2, :] = (inners, numpy.roll(inners, -1, axis=0))

        self.triangles = result.reshape((2*self.polygon.n, 3, 2))

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
