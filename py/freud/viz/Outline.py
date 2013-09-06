import numpy

from freud.shape import Polygon

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
        a Ntx3x2 numpy array of triangles and the inner field to a
        Polygon that is the interior polygon."""
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
        hs[concave] /= numpy.sin((dthetas[concave]-numpy.pi)/2)
        hs = hs.reshape((len(hs), 1))

        bisectors = ns - numpy.roll(ns, 1, axis=0)
        bisectors /= numpy.sqrt(numpy.sum(bisectors*bisectors, axis=1)).reshape((len(ns), 1))

        inners = self.polygon.vertices + hs.reshape((len(ns), 1))*bisectors
        self.inner = Polygon(inners)

        result = numpy.empty((2, self.polygon.n, 3, 2), dtype=numpy.float32)
        result[:, :, 0, :] = (self.polygon.vertices, inners)
        result[:, :, 1, :] = numpy.roll(self.polygon.vertices, -1, axis=0)
        result[:, :, 2, :] = (inners, numpy.roll(inners, -1, axis=0))

        self.triangles = result.reshape((2*self.polygon.n, 3, 2))
