import numpy
import re
from _freud import complement

## \package freud.complement
#
# Methods to compute shape pairings
#

# Mostly sure that my redefinitions of shapes have completely changed the angle finding stuff...no!

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

    # calculates a single match
    def single_match(self, rmax, s_dot_target, s_dot_tol, c_dot_target, c_dot_tol):
        # initializer
        smatch = complement(self.box, rmax, s_dot_target, s_dot_tol, c_dot_target, c_dot_tol)
        # create the list
        match_list = numpy.zeros(self.np, dtype=numpy.int32)
        # compute matches
        smatch.compute(match_list, self.positions, self.shape_angle, self.comp_angle)

        # get the number of matches
        nmatch = smatch.getNpair()

        return match_list, nmatch

    # the one ring of complementary finding

    def find_pairs(self,
                    base,
                    shape_type,
                    positions,
                    orientations,
                    types,
                    wv=1,
                    rmax=1.0,
                    s_dot_target=1.0,
                    s_dot_tol=0.1,
                    c_dot_target=1.0,
                    c_dot_tol=0.1):
        # base is the shape that we are working with. Int
        # s_ang = shape angles
        # c_ang = complementary angles
        # coded so that there are two different lists
        # wv is not needed for anything except hexes, and remember to grab/use
        # should be able to put a lot in from the params or whatever
        s_ang = numpy.copy(orientations)
        c_ang = numpy.copy(orientations)
        if (base == 5):
            # all pents are allophic, so no rotation, double pass, weirdness required
            # just in case
            self.update(positions, s_ang, c_ang)
            smatch = complement(self.box, rmax, s_dot_target, s_dot_tol, c_dot_target, c_dot_tol)
            match_list = numpy.zeros(self.np, dtype=numpy.int32)
            smatch.compute(match_list, self.positions, self.shape_angle, self.comp_angle)
            nmatch = smatch.getNpair()
        elif (base == 6):
            # hexagons are weird! Need special treatment
            if shape_type == bool(re.match(".split", "split")):
                # how many types are in here? only 1 type...I think
                c_ang = (c_ang[i] + numpy.pi) % (2.0 * numpy.pi)
                self.update(positions, s_ang, c_ang)
                smatch = complement(self.box, rmax, s_dot_target, s_dot_tol, c_dot_target, c_dot_tol)
                match_list = numpy.zeros(self.np, dtype=numpy.int32)
                smatch.compute(match_list, self.positions, self.shape_angle, self.comp_angle)
                nmatch = smatch.getNpair()
            if shape_type == bool(re.match(".comp", "comp")):
                # so now we need wv
                if (wv % 2 == 0):
                    # Need to check A-A', B-B', and A-B
                    # start with A-B
                    self.update(positions, s_ang, c_ang)
                    match_list0 = numpy.zeros(self.np, dtype=numpy.int32)
                    match_list1 = numpy.zeros(self.np, dtype=numpy.int32)
                    match_list = numpy.zeros(self.np, dtype=numpy.int32)
                    smatch0 = complement(self.box, rmax, s_dot_target, s_dot_tol, c_dot_target, c_dot_tol)
                    smatch0.compute(match_list0, self.positions, self.shape_angle, self.comp_angle)
                    nmatch0 = smatch.getNpair()
                    # Now for the others
                    c_ang = (c_ang[i] + numpy.pi) % (2.0 * numpy.pi)
                    self.update(positions, s_ang, c_ang)
                    smatch1 = complement(self.box, rmax, s_dot_target, s_dot_tol, c_dot_target, c_dot_tol)
                    match_list1 = numpy.zeros(self.np, dtype=numpy.int32)
                    smatch1.compute(match_list1, self.positions, self.shape_angle, self.comp_angle)
                    nmatch1 = smatch.getNpair()
                    for i in range(self.np):
                        if ((match_list0[i] == 1) or (match_list1[i] == 1)):
                            match_list[i] = 1
                        else:
                            match_list[i] = 0
                    nmatch = nmatch0 + nmatch1
                else:
                    self.update(positions, s_ang, c_ang)
                    match_list = numpy.zeros(self.np, dtype=numpy.int32)
                    smatch = complement(self.box, rmax, s_dot_target, s_dot_tol, c_dot_target, c_dot_tol)
                    smatch.compute(match_list, self.positions, self.shape_angle, self.comp_angle)
                    nmatch = smatch.getNpair()
        else:
            raise RuntimeError("incorrect shape/base spec'd")

        return match_list, nmatch


    # # compute matches for complementary particles
    # def complementary_match(self, positions, orientations, types,
    #                         rmax=1.0, s_dot_target=1.0, s_dot_tol=0.1, c_dot_target=1.0, c_dot_tol=0.1):
    #     # s_ang = shape angles
    #     # c_ang = complementary angles
    #     # coded so that there are two different lists
    #     s_ang = numpy.copy(orientations)
    #     c_ang = numpy.copy(orientations)
    #     # rotating angles for matching; these are most definitely wrong now
    #     # probably have to recode based on geometries
    #     for i in range(len(orientations)):
    #         if types[i] == 'A':
    #             s_ang[i] = (s_ang[i] + (numpy.pi / 2.0)) % (2.0 * numpy.pi)
    #             c_ang[i] = c_ang[i]
    #         if types[i] == 'B':
    #             s_ang[i] = (s_ang[i] + (numpy.pi / 2.0)) % (2.0 * numpy.pi)
    #             c_ang[i] = (c_ang[i] + numpy.pi) % (2.0 * numpy.pi)
    #     self.update(positions, s_ang, c_ang)
    #     smatch = complement(self.box, rmax, s_dot_target, s_dot_tol, c_dot_target, c_dot_tol)
    #     match_list = numpy.zeros(self.np, dtype=numpy.int32)
    #     smatch.compute(match_list, self.positions, self.shape_angle, self.comp_angle)
    #     nmatch = smatch.getNpair()
    #     return match_list, nmatch

    # def self_complementary_match(self, positions, orientations, types,
    #                         rmax=1.0, s_dot_target=1.0, s_dot_tol=0.1, c_dot_target=1.0, c_dot_tol=0.1):
    #     s_ang = numpy.copy(orientations)
    #     c_ang = numpy.copy(orientations)
    #     for i in range(len(orientations)):
    #         if types[i] == 'A':
    #             s_ang[i] = (s_ang[i] + (numpy.pi / 2.0)) % (2.0 * numpy.pi)
    #             c_ang[i] = c_ang[i]
    #         if types[i] == 'B':
    #             s_ang[i] = (s_ang[i] + (numpy.pi / 2.0)) % (2.0 * numpy.pi)
    #             c_ang[i] = (c_ang[i] + numpy.pi) % (2.0 * numpy.pi)
    #     self.update(positions, s_ang, c_ang)
    #     match0 = complement(self.box, rmax, s_dot_target, s_dot_tol, c_dot_target, c_dot_tol)
    #     match_list0 = numpy.zeros(self.np, dtype=numpy.int32)
    #     match0.compute(match_list0, self.positions, self.shape_angle, self.comp_angle)
    #     nmatch0 = smatch.getNpair()
    #     match1 = complement(self.box, rmax, (-1.0 * s_dot_target), s_dot_tol, c_dot_target, c_dot_tol)
    #     match_list1 = numpy.zeros(self.np, dtype=numpy.int32)
    #     match1.compute(match_list1, self.positions, self.shape_angle, self.comp_angle)
    #     nmatch1 = smatch.getNpair()
    #     match_list = numpy.zeros(self.np, dtype=numpy.float32)
    #     for i in range(self.np):
    #         if ((match_list0[i] == 1) or (match_list1[i] == 1)):
    #             match_list[i] = 1
    #         else:
    #             match_list[i] = 0
    #     nmatch = nmatch0 + nmatch1
    #     return match_list, nmatch
