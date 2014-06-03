## \package freud.pmft
#
# Methods to compute pair correlation function and pmft from point distributions.
#

import numpy

from _freud import PMFTXYZ
from _freud import PMFXY2D
from _freud import PMFTXYT2D

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

    # def compute(self, refPos=None, pos=None):
    def compute(self, refPos=None, refAng=None, pos=None, ang=None):
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
        if refAng is not None:
            self.refAng = refAng
        if ang is not None:
            self.ang = ang
        self.xArray = numpy.copy(self.pmftHandle.getX())
        self.yArray = numpy.copy(self.pmftHandle.getY())
        self.zArray = numpy.copy(self.pmftHandle.getZ())
        self.nBinsX = int(len(self.xArray))
        self.nBinsY = int(len(self.yArray))
        self.nBinsZ = int(len(self.zArray))
        pcfArray = numpy.zeros(shape=(self.nBinsZ, self.nBinsY, self.nBinsX), dtype=numpy.int32)
        self.pmftHandle.compute(pcfArray, self.refPos, self.refAng, self.pos, self.ang)
        self.pcfArray = numpy.copy(pcfArray)

class pmfXY2D(object):
    def __init__(self, box, maxX, maxY, dx, dy):
        super(pmfXY2D, self).__init__()
        self.box = box
        self.maxX = maxX
        self.maxY = maxY
        self.dx = dx
        self.dy = dy
        self.pmftHandle = PMFXY2D(self.box, self.maxX, self.maxY, self.dx, self.dy)

    # def compute(self, refPos=None, pos=None):
    def compute(self, refPos=None, refAng=None, pos=None, ang=None):
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
        if refAng is not None:
            self.refAng = refAng
        else:
            if self.refAng is None:
                raise RuntimeError("must input orientations")
        if ang is not None:
            self.ang = ang
        else:
            if self.ang is None:
                raise RuntimeError("must input orientations")
        self.xArray = numpy.copy(self.pmftHandle.getX())
        self.yArray = numpy.copy(self.pmftHandle.getY())
        self.nBinsX = int(len(self.xArray))
        self.nBinsY = int(len(self.yArray))
        pcfArray = numpy.zeros(shape=(self.nBinsY, self.nBinsX), dtype=numpy.int32)
        self.pmftHandle.compute(pcfArray, self.refPos, self.refAng, self.pos, self.ang)
        self.pcfArray = numpy.copy(pcfArray)

class pmftXYT2D(object):
    def __init__(self, box, maxX, maxY, maxT, dx, dy, dT):
        super(pmftXYT2D, self).__init__()
        self.box = box
        self.maxX = maxX
        self.maxY = maxY
        self.maxT = maxT
        self.dx = dx
        self.dy = dy
        self.dT = dT
        self.pmftHandle = PMFTXYT2D(self.box, self.maxX, self.maxY, self.maxT, self.dx, self.dy, self.dT)

    # def compute(self, refPos=None, pos=None):
    def compute(self, refPos=None, refAng=None, pos=None, ang=None):
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
        if refAng is not None:
            self.refAng = refAng
        else:
            if self.refAng is None:
                raise RuntimeError("must input orientations")
        if ang is not None:
            self.ang = ang
        else:
            if self.ang is None:
                raise RuntimeError("must input orientations")
        self.xArray = numpy.copy(self.pmftHandle.getX())
        self.yArray = numpy.copy(self.pmftHandle.getY())
        self.TArray = numpy.copy(self.pmftHandle.getT())
        self.nBinsX = int(len(self.xArray))
        self.nBinsY = int(len(self.yArray))
        self.nBinsT = int(len(self.TArray))
        pcfArray = numpy.zeros(shape=(self.nBinsT, self.nBinsY, self.nBinsX), dtype=numpy.int32)
        self.pmftHandle.compute(pcfArray, self.refPos, self.refAng, self.pos, self.ang)
        self.pcfArray = numpy.copy(pcfArray)
