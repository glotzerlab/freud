import numpy
import re
import multiprocessing
from _freud import pairing
from _freud import setNumThreads

## \package freud.pairing
#
# Methods to compute shape pairings

## Create a Pair object from a list of points, orientations, and shapes
#
# User must supply a box, max compute distance (this can/should be quite small). Number of threads for multithreading
# may be specified. To calculate, the user must supply a list of positions, two lists of orientations, as well as two
# dot product targets, and two dot product tolerances.
#
# two particles are matching if:
# \vec{r_i}: \; \text{position} \\
# \theta_s: \; \text{shape orientation angle} \\
# \theta_c: \; \text{complementary edge orientation angle} \\
# \hat{u}_s = \left( \cos \left( \theta_s \right), \; \sin \left( \theta_s \right) \right): \; \text{shape orientation unit vector} \\
# \hat{u}_c = \left( \cos \left( \theta_c \right), \; \sin \left( \theta_c \right) \right): \; \text{complementary edge orientation unit vector}
# \vec{r_{ij}} = \vec{r_j} - \vec{r_i} \\
# \hat{r_{ij}} = \frac{\vec{r_{ij}}}{|\vec{r_{ij}}|}
# |\vec{r_{ij}}| \leq d \\
# \hat{u}_{is} \cdot \hat{u}_{js} = s_{\text{target}} \\
# \hat{u}_{ic} \cdot \hat{u}_{jc} = c_{\text{target}} \\
# \hat{u}_{ic} \cdot \hat{r_{ij}} > 0
class Pair:
    ## Initialize Pair:
    # \param box The simulation box
    # \param rmax The max distance to search for pairings
    # \param nthreads Number of threads for tbb to use
    def __init__(self,
                 box,
                 rmax,
                 nthreads=None):
        self.box = box
        self.rmax = rmax
        if nthreads is not None:
            setNumThreads(int(nthreads))
        else:
            setNumThreads(multiprocessing.cpu_count())
        self.shape_orientations = None
        self.comp_orientations = None
        self.s_dot_target = None
        self.s_dot_tol = None
        self.c_dot_target = None
        self.c_dot_tol = None

    ## Update relevant variables. Mainly called through find_pairs
    # \params positions The positions of the particles
    # \params shape_orientations The orientation of the shape itself
    # \params comp_orientations The orientation of the complementary interface
    # \params s_dot_target The target dot product for the shape vectors
    # \params s_dot_tol The tolerance for the shape dot product
    # \params c_dot_target The target dot product for the complementary vectors
    # \params c_dot_tol The tolerance for the complementary dot product
    def update(self,
               positions=None,
               shape_orientations=None,
               comp_orientations=None,
               s_dot_target=None,
               s_dot_tol=None,
               c_dot_target=None,
               c_dot_tol=None):
        if positions is not None:
            self.positions = numpy.copy(positions)
        if shape_orientations is not None:
            self.shape_orientations = numpy.copy(shape_orientations)
        if comp_orientations is not None:
            self.comp_orientations = numpy.copy(comp_orientations)
        if s_dot_target is not None:
            self.s_dot_target = s_dot_target
        if s_dot_tol is not None:
            self.s_dot_tol = s_dot_tol
        if c_dot_target is not None:
            self.c_dot_target = c_dot_target
        if c_dot_tol is not None:
            self.c_dot_tol = c_dot_tol
        self.np = len(self.positions)

    ## Do the actual calculation
    # \params positions The positions of the particles
    # \params shape_orientations The orientation of the shape itself
    # \params comp_orientations The orientation of the complementary interface
    # \params s_dot_target The target dot product for the shape vectors
    # \params s_dot_tol The tolerance for the shape dot product
    # \params c_dot_target The target dot product for the complementary vectors
    # \params c_dot_tol The tolerance for the complementary dot product
    def find_pairs(self,
                    positions,
                    shape_orientations,
                    comp_orientations,
                    s_dot_target=None,
                    s_dot_tol=None,
                    c_dot_target=None,
                    c_dot_tol=None):
        match_list = numpy.zeros(shape=len(positions), dtype=numpy.int32)
        self.update(positions,
                    shape_orientations,
                    comp_orientations,
                    s_dot_target,
                    s_dot_tol,
                    c_dot_target,
                    c_dot_tol)
        if self.shape_orientations is None:
            raise RuntimeError("no orientations specified")
        if self.comp_orientations is None:
            raise RuntimeError("no orientations specified")
        if self.s_dot_target is None:
            raise RuntimeError("no shape dot product target specified")
        if self.c_dot_target is None:
            raise RuntimeError("no complementary dot product target specified")
        if self.s_dot_tol is None:
            raise RuntimeError("no shape dot product tol specified")
        if self.c_dot_tol is None:
            raise RuntimeError("no complementary dot product tol specified")
        smatch = pairing(self.box, self.rmax, self.s_dot_target, self.s_dot_tol, self.c_dot_target, self.c_dot_tol)
        smatch.compute(match_list, self.positions, self.shape_orientations, self.comp_orientations)
        nmatch = numpy.sum(match_list) / 2.0

        return match_list, nmatch
