import numpy
from _freud import complement

## \package freud.complement
#
# Methods to compute shape pairings
#

class Pair:
    ## Create a Pair object from a list of points, orientations, and shapes
    def __init__(self, box, shapes, ntypes, positions=None, orientations=None, types=None):
        self.box = box
        # shapes is a dict
        self.shapes = shapes
        self.ntypes = ntypes
        self.points = {}
        self.cavities = {}

    def update(self, positions, orientations, types):
        self.positions = positions
        self.orientations = orientations
        self.types = types
        if len(self.positions) != len(self.orientations):
            raise RuntimeError("length of positions and orientations must be the same")
        self.np = len(self.positions)

# points must be set in the same order as their corresponding cavity for perfect match
    def set_points(self, shape, points):
        tmp_shape = self.shapes[shape]
        point_list = []
        for point in points:
            if point < len(tmp_shape):
                point_list.append(point)
            else:
                raise RuntimeError("Impossible Point Specified")
        self.points[shape] = point_list

    def set_cavities(self, shape, cavities):
        tmp_shape = self.shapes[shape]
        cavity_list = []
        for cavity in cavities:
            if cavity < len(tmp_shape):
                cavity_list.append(cavity)
            else:
                raise RuntimeError("Impossible Cavity Specified")
        self.cavities[shape] = cavity_list

    def single_match(self, rmax, ref_type, check_type):
        if not len(self.points) == len(self.cavities):
            raise RuntimeError("single_match can't be called on unequal sized lists")
        smatch = complement(self.box, rmax)
        match_list = numpy.zeros(self.np, dtype=numpy.int32)
        type_list = numpy.zeros(self.np, dtype=numpy.int32)
        cnt = 0
        for t in self.types:
            if t == 'A':
                type_list[cnt] = 0
            elif t == 'B':
                type_list[cnt] = 1
            cnt += 1

        if self.ntypes == 1:
            shape_list = numpy.array([self.shapes['A']], dtype=numpy.float32)
        elif self.ntypes == 2:
            shape_list = numpy.array([self.shapes['A'], self.shapes['B']], dtype=numpy.float32)
        # shape_list = numpy.array([self.shapes[i] for i in range(len(self.shapes))])

        if ref_type == 'A':
            ref_num = numpy.array([0], dtype=numpy.int32)
        elif ref_type == 'B':
            ref_num = numpy.array([1], dtype=numpy.int32)

        if check_type == 'A':
            check_num = numpy.array([0], dtype=numpy.int32)
        elif check_type == 'B':
            check_num = numpy.array([1], dtype=numpy.int32)

        ref_verts = numpy.array(self.points[ref_type], dtype=numpy.int32)

        check_verts = numpy.array(self.cavities[check_type], dtype=numpy.int32)

        smatch.compute(match_list, self.positions, type_list, self.orientations, shape_list, ref_num, check_num, ref_verts, check_verts)

        nmatch = smatch.getNpair()

        return match_list, nmatch
        # No idea how to work with multi_match right now
