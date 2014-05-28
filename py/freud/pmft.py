## \package freud.pmft
#
# Methods to compute pair correlation function and pmft from point distributions.
#

import numpy

from _freud import PMFTXYZ

class pmftXYZ(object):
    def __init__(self, box, maxX, maxY, maxZ, dx, dy, dz):
        super(pmftXYZ, self).__init__()
        self.box = box
        self.maxX = maxX
        self.maxY = maxY
        self.maxZ = maxZ
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.pmftHandle = PMFTXYZ(self.box, self.maxX, self.maxY, self.maxZ, self.dx, self.dy, self.dz)

    def compute(self, refPos=None, pos=None):
        if refPos is not None:
            self.refPos = refPos
        else:
            if self.refPos is None:
                raise RuntimeError("must input positions")
        if pos is not None:
            self.pos = pos
        else:
            if self.pos is None:
                raise RuntimeError("must input positions")
        self.pmftHandle.compute(self.refPos, self.pos)
        self.pcfRaw = numpy.copy(self.pmftHandle.getPCF())
        self.xArray = numpy.copy(self.pmftHandle.getX())
        self.yArray = numpy.copy(self.pmftHandle.getY())
        self.zArray = numpy.copy(self.pmftHandle.getZ())
        # reshape pcf array
        # calculate shape
        self.nBinsX = int(len(self.xArray))
        self.nBinsY = int(len(self.yArray))
        self.nBinsZ = int(len(self.zArray))
        self.pcfArray = numpy.reshape(self.pcfRaw, (self.nBinsX, self.nBinsY, self.nBinsZ))
        # should add in a check

    def computePMFT(self):
        # check that pcf is calculated
        if self.pcfArray is None:
            raise RuntimeError("must compute pcf first")
        self.pmftArray = numpy.copy(self.pcfArray)
        # have to do this idiotically because neg log
        # self.pmftArray = -1.0 * numpy.log(self.pmftArray)
        # pmftShape = numpy.shape(self.pmftArray)
        # for i in range(pmftShape[0]):
        #     for j in range(pmftShape[1]):
        #         for k in range(pmftShape[2]):
        #             if self.pmftArray[i][j][k] == 0:
        #                 self.pmftArray[i][j][k] = 0
        #             else:
        #                 self.pmftArray[i][j][k] = -1.0 * numpy.log(self.pmftArray[i][j][k])
