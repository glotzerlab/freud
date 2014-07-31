## \package freud.pmft
#
# Methods to compute pair correlation function and pmft from point distributions.
#

import numpy

from _freud import Bootstrap

# I need to think about how I can handle multiple input types...

## Compute the bootstrap analysis
# might need to change the name of this
class bootstrap(object):
    ## Initialize the bootstrap object:
    # \param dataArr The data array to compute the bootstrap on
    # \param nBootstrap number of bootstraps to compute
    def __init__(self, dataArr, nBootstrap):
        super(bootstrap, self).__init__()
        self.dataArr = dataArr
        self.nBootstrap = nBootstrap
        # collapse the array and turn into a cumulative array
        self.dataFlat = numpy.copy(self.dataArr.flatten())
        self.dataCum = numpy.copy(self.dataFlat)
        self.arrSize = 1
        for i in self.dataArr.shape:
            self.arrSize *= i
        assert len(dataCum) == myProd
        # done in python as this only needs done once; probably a better way to do
        for i in range(1, myProd):
            self.dataCum[i] += self.dataCum[i-1]
        self.bootstrapArray = numpy.zeros(shape=(nBootstrap, myProd), dtype=numpy.int32)
        self.bootstrapAVG = numpy.zeros(shape=self.dataArr.shape, dtype=numpy.float32)
        self.bootstrapSTD = numpy.zeros(shape=self.dataArr.shape, dtype=numpy.float32)
        self.bootstrapRatio = numpy.copy(1.0 / self.dataArr.astype(dtype=numpy.float32))
        self.nPoints = self.dataCum[-1]
        self.bootstrapHandle = Bootstrap(self.nBootstrap, self.nPoints, self.arrSize)

    ## Compute the aniso pmf for a given set of points (one traj frame)
    # currently seg faulting on some of the new stuff. Need to find the faults
    def compute(self):
        self.bootstrapHandle.compute(self.bootstrapArray, self.bootstrapAVG, self.bootstrapSTD, self.bootstrapRatio, self.dataCum)
