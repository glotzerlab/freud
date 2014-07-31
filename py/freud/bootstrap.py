## \package freud.pmft
#
# Methods to compute pair correlation function and pmft from point distributions.
#
import sys
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
        # check that dataArr is a uint32
        super(bootstrap, self).__init__()
        self.dataArr = numpy.copy(dataArr)
        self.nBootstrap = nBootstrap
        # collapse the array and turn into a cumulative array
        self.dataFlat = numpy.copy(self.dataArr.flatten())
        self.dataCum = numpy.copy(self.dataFlat)
        self.arrSize = 1
        for i in self.dataArr.shape:
            self.arrSize *= i
        assert len(self.dataCum) == self.arrSize
        # done in python as this only needs done once; probably a better way to do
        for i in range(1, self.arrSize):
            self.dataCum[i] += self.dataCum[i-1]
        self.nPoints = int(self.dataCum[-1])
        self.bootstrapHandle = Bootstrap(self.nBootstrap, self.nPoints, self.arrSize)

    ## Compute the aniso pmf for a given set of points (one traj frame)
    # currently seg faulting on some of the new stuff. Need to find the faults
    def compute(self):
        bootstrapAVG = numpy.zeros(shape=self.dataArr.shape, dtype=numpy.float32)
        bootstrapSTD = numpy.zeros(shape=self.dataArr.shape, dtype=numpy.float32)
        bootstrapRatio = numpy.zeros(shape=self.dataArr.shape, dtype=numpy.float32)
        self.bootstrapHandle.compute(bootstrapAVG, bootstrapSTD, bootstrapRatio, self.dataCum, self.dataFlat)
        self.bootstrapAVG = numpy.copy(bootstrapAVG)
        self.bootstrapSTD = numpy.copy(bootstrapSTD)
        self.bootstrapRatio = numpy.copy(bootstrapRatio)
