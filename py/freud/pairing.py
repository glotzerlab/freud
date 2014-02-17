import numpy
import re
from _freud import pairing

## \package freud.pairing
#
# Methods to compute shape pairings

class Pair:
    ## Create a Pair object from a list of points, orientations, and shapes
    def __init__(self, box):
        self.box = box

    # method that allows for the positions, angles to be changed
    def update(self, positions, shape_angle, comp_angle):
        self.positions = positions
        self.shape_angle = shape_angle
        self.comp_angle = comp_angle
        self.np = len(self.positions)

    # the one ring of complementary finding
    # all preconditioning happens elsewhere
    def find_pairs(self,
                    match_list,
                    nmatch,
                    positions,
                    s_orientations,
                    c_orientations,
                    rmax=1.0,
                    s_dot_target=1.0,
                    s_dot_tol=0.1,
                    c_dot_target=1.0,
                    c_dot_tol=0.1):
        # base is the shape that we are working with. Int
        # s_ang = shape angles
        # c_ang = complementary angles
        # coded so that there are two different lists
        s_ang = numpy.copy(s_orientations)
        c_ang = numpy.copy(c_orientations)
        self.update(positions, s_ang, c_ang)
        smatch = pairing(self.box, rmax, s_dot_target, s_dot_tol, c_dot_target, c_dot_tol)
        smatch.compute(match_list, self.positions, self.shape_angle, self.comp_angle)
        nmatch = smatch.getNpair()

        return match_list, nmatch
