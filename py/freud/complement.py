import numpy
from _freud import complement

## \package freud.complement
#
# Methods to compute shape pairings
#

class Pair:
    ## Create a Pair object from a list of points, orientations, and shapes
    def __init__(self, box):
        self.box = box

    def update(self, positions, shape_angle, comp_angle):
        self.positions = positions
        self.shape_angle = shape_angle
        self.comp_angle = comp_angle
        self.np = len(self.positions)

    def single_match(self, rmax, s_dot_target, s_dot_tol, c_dot_target, c_dot_tol):
        smatch = complement(self.box, rmax, s_dot_target, s_dot_tol, c_dot_target, c_dot_tol)
        match_list = numpy.zeros(self.np, dtype=numpy.int32)
        smatch.compute(match_list, self.positions, self.shape_angle, self.comp_angle)

        nmatch = smatch.getNpair()

        return match_list, nmatch
    
    def complementary_match(self, positions, orientations, types,
                            rmax=1.0, s_dot_target=1.0, s_dot_tol=0.1, c_dot_target=1.0, c_dot_tol=0.1):
        s_ang = numpy.copy(orientations)
        c_ang = numpy.copy(orientations)
        for i in range(len(orientations)):
            if types[i] == 'A':
                s_ang[i] = (s_ang[i] + (numpy.pi / 2.0)) % (2.0 * numpy.pi)
                c_ang[i] = c_ang[i]
            if types[i] == 'B':
                s_ang[i] = (s_ang[i] + (numpy.pi / 2.0)) % (2.0 * numpy.pi)
                c_ang[i] = (c_ang[i] + numpy.pi) % (2.0 * numpy.pi)
        self.update(positions, s_ang, c_ang)
        smatch = complement(self.box, rmax, s_dot_target, s_dot_tol, c_dot_target, c_dot_tol)
        match_list = numpy.zeros(self.np, dtype=numpy.int32)
        smatch.compute(match_list, self.positions, self.shape_angle, self.comp_angle)
        nmatch = smatch.getNpair()
        return match_list, nmatch

    def self_complementary_match(self, positions, orientations, types,
                            rmax=1.0, s_dot_target=1.0, s_dot_tol=0.1, c_dot_target=1.0, c_dot_tol=0.1):
        s_ang = numpy.copy(orientations)
        c_ang = numpy.copy(orientations)
        for i in range(len(orientations)):
            if types[i] == 'A':
                s_ang[i] = (s_ang[i] + (numpy.pi / 2.0)) % (2.0 * numpy.pi)
                c_ang[i] = c_ang[i]
            if types[i] == 'B':
                s_ang[i] = (s_ang[i] + (numpy.pi / 2.0)) % (2.0 * numpy.pi)
                c_ang[i] = (c_ang[i] + numpy.pi) % (2.0 * numpy.pi)
        self.update(positions, s_ang, c_ang)
        match0 = complement(self.box, rmax, s_dot_target, s_dot_tol, c_dot_target, c_dot_tol)
        match_list0 = numpy.zeros(self.np, dtype=numpy.int32)
        match0.compute(match_list0, self.positions, self.shape_angle, self.comp_angle)
        nmatch0 = smatch.getNpair()
        match1 = complement(self.box, rmax, (-1.0 * s_dot_target), s_dot_tol, c_dot_target, c_dot_tol)
        match_list1 = numpy.zeros(self.np, dtype=numpy.int32)
        match1.compute(match_list1, self.positions, self.shape_angle, self.comp_angle)
        nmatch1 = smatch.getNpair()
        match_list = numpy.zeros(self.np, dtype=numpy.float32)
        for i in range(self.np):
            if ((match_list0[i] == 1) or (match_list1[i] == 1)):
                match_list[i] = 1
            else:
                match_list[i] = 0
        nmatch = nmatch0 + nmatch1
        return match_list, nmatch
