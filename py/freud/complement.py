import numpy
from _freud import complement

## \package freud.complement
#
# Methods to compute shape pairings
#

class Pair:
    ## Create a Pair object from a list of points, orientations, and shapes
    def __init__(self, box, positions=None, orientations=None):
        self.box = box

    def update(self, positions, orientations):
        self.positions = positions
        self.orientations = orientations
        self.np = len(self.positions)

    def single_match(self, rmax, dot_target, dot_tol):
        smatch = complement(self.box, rmax, dot_target, dot_tol)
        match_list = numpy.zeros(self.np, dtype=numpy.int32)
        smatch.compute(match_list, self.positions, self.orientations)

        nmatch = smatch.getNpair()

        return match_list, nmatch
        # No idea how to work with multi_match right now
