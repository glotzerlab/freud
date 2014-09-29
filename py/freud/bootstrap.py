## \package freud.pmft
#
# Methods to compute pair correlation function and pmft from point distributions.
#
import sys
import numpy

from _freud import Bootstrap

## Compute the bootstrap analysis
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
        self.arrSize = 1
        for i in self.dataArr.shape:
            self.arrSize *= i
        assert len(self.dataFlat) == self.arrSize
        # done in python as this only needs done once; probably a better way to do
        # print("getting ready to create bootstrap object")
        self.bootstrapHandle = Bootstrap(self.nBootstrap, self.dataFlat)

    ## Compute the aniso pmf for a given set of points (one traj frame)
    def compute(self):
        self.bootstrapHandle.compute()
        self.bootstrapArray = self.bootstrapHandle.getBootstrap()
        # contains the flat version of the array
        self.bootstrapAVG = self.bootstrapHandle.getAVG()
        self.dataAVG = numpy.copy(self.bootstrapAVG.reshape(self.dataArr.shape))
        self.bootstrapSTD = self.bootstrapHandle.getSTD()
        self.dataSTD = numpy.copy(self.bootstrapSTD.reshape(self.dataArr.shape))
        self.bootstrapERR = self.bootstrapHandle.getERR()
        self.dataERR = numpy.copy(self.bootstrapERR.reshape(self.dataArr.shape))
