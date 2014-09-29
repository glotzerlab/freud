import numpy
import re
from _freud import pairing

## \package freud.pairing
#
# Methods to compute shape pairings

## Create a Pair object from a list of points and orientations
#
# Two particles are matching if:
# they are the two nearest neighbors and the following are true:
# \vec{r_i}: \; \text{position} \\
# \theta: \; \text{particle orientation} \\
# \theta_c: \; \text{complementary edge orientation angle in local orientation} \\
# \hat{u}_c = e^{i \theta_c} = \cos \left( \theta_c \right) + i \sin \left( \theta_c \right): \; \text{complementary edge orientation unit vector}
# \vec{r_{ij}} = \vec{r_j} - \vec{r_i} \\
# \hat{r_{ij}} = \frac{\vec{r_{ij}}}{|\vec{r_{ij}}|}
# |\vec{r_{ij}}| \leq d \\
# \hat{u}_{ic} \cdot \hat{r}_{ij} = 1 \\
# \hat{u}_{jc} \cdot \hat{r}_{ji} = 1 \\
class Pair2D:
    ## Initialize Pair:
    # \param box The simulation box
    # \param rmax The max distance to search for nearest neighbors
    # \param k The number of nearest neighbors to check
    # \params cDotTol The tolerance for the complementary dot product as an angle, in radians
    def __init__(self,box,rmax,k,cDotTol):
        super(Pair2D, self).__init__()
        self.box = box
        self.rmax = rmax
        self.k = int(k)
        self.cDotTol = cDotTol
        self.pairHandle = pairing(self.box, self.rmax, self.k, self.cDotTol)

    ## Update relevant variables. Mainly called through compute
    # \params positions The positions of the particles
    # \params orientations The orientation of the particle
    # \params compOrientations The orientations of potential complementary interfaces
    def update(self,
               positions=None,
               orientations=None,
               compOrientations=None):
        if positions is not None:
            self.positions = numpy.copy(positions)
        if orientations is not None:
            self.orientations = numpy.copy(orientations)
        if compOrientations is not None:
            self.compOrientations = numpy.copy(compOrientations)
        self.np = len(self.positions)

    ## Do the actual calculation
    # \params positions The positions of the particles
    # \params orientations The orientation of the shape itself
    # \params compOrientations The orientation of the complementary interface
    def compute(self,
                positions,
                orientations,
                compOrientations):
        self.update(positions,
                    orientations,
                    compOrientations)
        if self.orientations is None:
            raise RuntimeError("no orientations specified")
        if self.compOrientations is None:
            raise RuntimeError("no complementary orientations specified")
        self.pairHandle.compute(self.positions, self.orientations, self.compOrientations)
        self.matchList = self.pairHandle.getMatch()
        self.pairList = self.pairHandle.getPair()
        self.nMatch = numpy.sum(self.matchList)
