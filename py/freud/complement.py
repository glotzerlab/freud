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
        # No idea how to work with multi_match right now
