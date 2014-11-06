## \package freud.pmft
#
# Methods to compute pair correlation function and pmft from point distributions.
#

import numpy
import time

from _freud import PMFXYZ
from _freud import PMFXY2D
from _freud import PMFTXYT2D
from _freud import PMFTXYTP2D
from _freud import PMFTXYTM2D
from _freud import PMFTRPM

## Computes the 3D anisotropic potential of mean force
# While the direct interface to c++ is made available as part of the self.pmftHandle,
# you should avoid doing this unless you know exactly what you are doing
class pmfXYZ(object):
    ## Initialize the pmftXYZ object:
    # \param box The simulation box from freud
    # \param maxX The maximum distance to consider in the x-direction (both + and -)
    # \param maxY The maximum distance to consider in the y-direction (both + and -)
    # \param maxZ The maximum distance to consider in the z-direction (both + and -)
    # \param dx The bin size in the x-direction
    # \param dy The bin size in the y-direction
    # \param dz The bin size in the z-direction
    def __init__(self, maxX, maxY, maxZ, dx, dy, dz):
        super(pmfXYZ, self).__init__()
        self.maxX = maxX
        self.maxY = maxY
        self.maxZ = maxZ
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.pmfHandle = PMFXYZ(self.maxX, self.maxY, self.maxZ, self.dx, self.dy, self.dz)
        self.xArray = self.pmfHandle.getX()
        self.yArray = self.pmfHandle.getY()
        self.zArray = self.pmfHandle.getZ()
        self.nBinsX = int(len(self.xArray))
        self.nBinsY = int(len(self.yArray))
        self.nBinsZ = int(len(self.zArray))

    ## Compute the aniso pmf for a given set of points (one traj frame)
    # \param refPos Reference point to consider
    # \param refOrientations Reference orientation to consider as quaternion
    # \param pos points to consider
    # \param orientations orientations to consider as quaternion
    # \param faceQuaternions orientations to rotate after bringing into local coordinates
    def compute(self, box, refPos=None, refOrientations=None, pos=None, orientations=None, faceQuaternions=None):
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
        if refOrientations is not None:
            self.refOrientations = refOrientations
        if orientations is not None:
            self.orientations = orientations
        if faceQuaternions is not None:
            # change this check so it doesn't barf
            faceShape = faceQuaternions.shape
            if faceShape[0] != len(self.refPos):
                raise RuntimeError("please supply the same number of quaternion sets as ref particles")
            self.nSets = faceShape[1]
            for i in faceQuaternions:
                mySet = len(i)
                if self.nSets != mySet:
                    raise RuntimeError("the same number of face quaternions needs to be supplied for all shapes")
            if faceShape[2] != 4:
                raise RuntimeError("quaternions require 4 values")
            self.faceQuaternions = faceQuaternions
        else:
            # create a unit quaternion
            self.faceQuaternions = numpy.zeros(shape=(len(self.refPos), 1, 4), dtype=numpy.float32)
            self.faceQuaternions[:,:,0] = 1.0
        self.pmfHandle.compute(box, self.refPos, self.refOrientations, self.pos, self.orientations, self.faceQuaternions)

    ## Calculate the PMF from the PCF. This has the side-effect of also populating the self.pcfArray
    # in addition to self.pmfArray
    # after calling calcPMF(), you can access the results via:
    # self.pcfArray for the positional correlation function
    # self.avgOccupancy for the average bin occupancy (useful for sanity checking)
    # self.pmfArray will give the pmf
    # NOTE: self.pcfArray is of type numpy.uint32; if averaging multiple frames,
    # you need to recast as floats: self.pcfArray.astype(numpy.float32)
    def calcPMF(self):
        self.pcfArray = self.pmfHandle.getPCF()
        self.pcfArray = self.pcfArray.reshape((self.nBinsZ, self.nBinsY, self.nBinsX))
        self.avgOccupancy = numpy.sum(numpy.sum(numpy.sum(self.pcfArray))) / (self.nBinsX * self.nBinsY * self.nBinsZ)
        self.pmfArray = -numpy.log(numpy.copy(self.pcfArray))

    ## Reset the PCF array
    # Call when you want to zero out the pcf array in C
    # This should be done if you need to change the types of particles being compared
    def reset(self):
        self.pmfHandle.resetPMF()

## Computes the 2D anisotropic potential of mean force
# While the direct interface to c++ is made available as part of the self.pmftHandle,
# you should avoid doing this unless you know exactly what you are doing
class pmfXY2D(object):
    ## Initialize the pmfXY2D object:
    # \param box The simulation box from freud
    # \param maxX The maximum distance to consider in the x-direction (both + and -)
    # \param maxY The maximum distance to consider in the y-direction (both + and -)
    # \param dx The bin size in the x-direction
    # \param dy The bin size in the y-direction
    def __init__(self, maxX, maxY, dx, dy):
        super(pmfXY2D, self).__init__()
        self.maxX = maxX
        self.maxY = maxY
        self.dx = dx
        self.dy = dy
        self.pmfHandle = PMFXY2D(self.maxX, self.maxY, self.dx, self.dy)
        self.xArray = self.pmfHandle.getX()
        self.yArray = self.pmfHandle.getY()
        self.nBinsX = int(len(self.xArray))
        self.nBinsY = int(len(self.yArray))

    ## Compute the aniso pmf for a given set of points (one traj frame)
    # \param refPos Reference point to consider
    # \param refAng Reference angles to consider as floats
    # \param pos Points to consider
    # \param ang Angles to consider as floats
    def compute(self, box, refPos=None, refAng=None, pos=None, ang=None):
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
        self.pmfHandle.compute(box, self.refPos, self.refAng, self.pos, self.ang)

    ## Calculate the PMF from the PCF. This has the side-effect of also populating the self.pcfArray
    # in addition to self.pmfArray
    # after calling calcPMF(), you can access the results via:
    # self.pcfArray for the positional correlation function
    # self.avgOccupancy for the average bin occupancy (useful for sanity checking)
    # self.pmfArray will give the pmf
    # NOTE: self.pcfArray is of type numpy.uint32; if averaging multiple frames,
    # you need to recast as floats: self.pcfArray.astype(numpy.float32)
    def calcPMF(self):
        self.pcfArray = self.pmfHandle.getPCF()
        self.pcfArray = self.pcfArray.reshape((self.nBinsY, self.nBinsX))
        self.avgOccupancy = numpy.sum(numpy.sum(self.pcfArray)) / (self.nBinsX * self.nBinsY)
        self.pmfArray = -numpy.log(numpy.copy(self.pcfArray))

    ## Reset the PCF array
    # Call when you want to zero out the pcf array in C
    # This should be done if you need to change the types of particles being compared
    def reset(self):
        self.pmfHandle.resetPMF()

## Computes the 2D anisotropic potential of mean force and torque
class pmftXYT2D(object):
    ## Initialize the pmftXYT2D object:
    # \param box The simulation box from freud
    # \param maxX The maximum distance to consider in the x-direction (both + and -)
    # \param maxY The maximum distance to consider in the y-direction (both + and -)
    # \param maxT The maximum angle to consider (both + and -)
    # \param dx The bin size in the x-direction
    # \param dy The bin size in the y-direction
    # \param dT The angle bin size
    def __init__(self, maxX, maxY, maxT, dx, dy, dT):
        super(pmftXYT2D, self).__init__()
        self.maxX = maxX
        self.maxY = maxY
        self.maxT = maxT
        self.dx = dx
        self.dy = dy
        self.dT = dT
        self.pmftHandle = PMFTXYT2D(self.maxX, self.maxY, self.maxT, self.dx, self.dy, self.dT)
        self.xArray = self.pmfHandle.getX()
        self.yArray = self.pmfHandle.getY()
        self.TArray = self.pmfHandle.getT()
        self.nBinsX = int(len(self.xArray))
        self.nBinsY = int(len(self.yArray))
        self.nBinsT = int(len(self.TArray))

    ## Compute the pmft for a given set of points (one traj frame)
    # \param refPos Reference point to consider
    # \param refAng Reference angles to consider as floats
    # \param pos Points to consider
    # \param ang Angles to consider as floats
    def compute(self, box, refPos=None, refAng=None, pos=None, ang=None):
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
        self.pmftHandle.compute(box, self.refPos, self.refAng, self.pos, self.ang)

    ## Calculate the PMF from the PCF. This has the side-effect of also populating the self.pcfArray
    # in addition to self.pmfArray
    # after calling calcPMF(), you can access the results via:
    # self.pcfArray for the positional correlation function
    # self.avgOccupancy for the average bin occupancy (useful for sanity checking)
    # self.pmfArray will give the pmf
    # NOTE: self.pcfArray is of type numpy.uint32; if averaging multiple frames,
    # you need to recast as floats: self.pcfArray.astype(numpy.float32)
    def calcPMF(self):
        self.pcfArray = self.pmfHandle.getPCF()
        self.pcfArray = self.pcfArray.reshape((self.nBinsT, self.nBinsY, self.nBinsX))
        self.avgOccupancy = numpy.sum(numpy.sum(numpy.sum(self.pcfArray))) / (self.nBinsX * self.nBinsY * self.nBinsT)
        self.pmfArray = -numpy.log(numpy.copy(self.pcfArray))

    ## Reset the PCF array
    # Call when you want to zero out the pcf array in C
    # This should be done if you need to change the types of particles being compared
    def reset(self):
        self.pmfHandle.resetPMF()

## Computes the 2D anisotropic potential of mean force and torque using \theta = \phi_1 + \phi_2
class pmftXYTP2D(object):
    ## Initialize the pmftXYT2D object:
    # \param box The simulation box from freud
    # \param maxX The maximum distance to consider in the x-direction (both + and -)
    # \param maxY The maximum distance to consider in the y-direction (both + and -)
    # \param maxT The maximum angle to consider (both + and -)
    # \param dx The bin size in the x-direction
    # \param dy The bin size in the y-direction
    # \param dT The angle bin size
    def __init__(self, maxX, maxY, maxT, dx, dy, dT):
        super(pmftXYTP2D, self).__init__()
        self.maxX = maxX
        self.maxY = maxY
        self.maxT = maxT
        self.dx = dx
        self.dy = dy
        self.dT = dT
        self.pmftHandle = PMFTXYTP2D(self.maxX, self.maxY, self.maxT, self.dx, self.dy, self.dT)
        self.xArray = self.pmfHandle.getX()
        self.yArray = self.pmfHandle.getY()
        self.TArray = self.pmfHandle.getT()
        self.nBinsX = int(len(self.xArray))
        self.nBinsY = int(len(self.yArray))
        self.nBinsT = int(len(self.TArray))

    ## Compute the pmft for a given set of points (one traj frame)
    # \param refPos Reference point to consider
    # \param refAng Reference angles to consider as floats
    # \param pos Points to consider
    # \param ang Angles to consider as floats
    def compute(self, box, refPos=None, refAng=None, pos=None, ang=None):
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
        self.pmftHandle.compute(box, self.refPos, self.refAng, self.pos, self.ang)

    ## Calculate the PMF from the PCF. This has the side-effect of also populating the self.pcfArray
    # in addition to self.pmfArray
    # after calling calcPMF(), you can access the results via:
    # self.pcfArray for the positional correlation function
    # self.avgOccupancy for the average bin occupancy (useful for sanity checking)
    # self.pmfArray will give the pmf
    # NOTE: self.pcfArray is of type numpy.uint32; if averaging multiple frames,
    # you need to recast as floats: self.pcfArray.astype(numpy.float32)
    def calcPMF(self):
        self.pcfArray = self.pmfHandle.getPCF()
        self.pcfArray = self.pcfArray.reshape((self.nBinsT, self.nBinsY, self.nBinsX))
        self.avgOccupancy = numpy.sum(numpy.sum(numpy.sum(self.pcfArray))) / (self.nBinsX * self.nBinsY * self.nBinsT)
        self.pmfArray = -numpy.log(numpy.copy(self.pcfArray))

    ## Reset the PCF array
    # Call when you want to zero out the pcf array in C
    # This should be done if you need to change the types of particles being compared
    def reset(self):
        self.pmfHandle.resetPMF()

## Computes the 2D anisotropic potential of mean force and torque using \theta = \phi_1 - \phi_2
class pmftXYTM2D(object):
    ## Initialize the pmftXYT2D object:
    # \param box The simulation box from freud
    # \param maxX The maximum distance to consider in the x-direction (both + and -)
    # \param maxY The maximum distance to consider in the y-direction (both + and -)
    # \param maxT The maximum angle to consider (both + and -)
    # \param dx The bin size in the x-direction
    # \param dy The bin size in the y-direction
    # \param dT The angle bin size
    def __init__(self, maxX, maxY, maxT, dx, dy, dT):
        super(pmftXYTM2D, self).__init__()
        self.maxX = maxX
        self.maxY = maxY
        self.maxT = maxT
        self.dx = dx
        self.dy = dy
        self.dT = dT
        self.pmftHandle = PMFTXYTM2D(self.maxX, self.maxY, self.maxT, self.dx, self.dy, self.dT)
        self.xArray = self.pmfHandle.getX()
        self.yArray = self.pmfHandle.getY()
        self.TArray = self.pmfHandle.getT()
        self.nBinsX = int(len(self.xArray))
        self.nBinsY = int(len(self.yArray))
        self.nBinsT = int(len(self.TArray))

    ## Compute the pmft for a given set of points (one traj frame)
    # \param refPos Reference point to consider
    # \param refAng Reference angles to consider as floats
    # \param pos Points to consider
    # \param ang Angles to consider as floats
    def compute(self, box, refPos=None, refAng=None, pos=None, ang=None):
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
        self.pmftHandle.compute(box, self.refPos, self.refAng, self.pos, self.ang)

    ## Calculate the PMF from the PCF. This has the side-effect of also populating the self.pcfArray
    # in addition to self.pmfArray
    # after calling calcPMF(), you can access the results via:
    # self.pcfArray for the positional correlation function
    # self.avgOccupancy for the average bin occupancy (useful for sanity checking)
    # self.pmfArray will give the pmf
    # NOTE: self.pcfArray is of type numpy.uint32; if averaging multiple frames,
    # you need to recast as floats: self.pcfArray.astype(numpy.float32)
    def calcPMF(self):
        self.pcfArray = self.pmfHandle.getPCF()
        self.pcfArray = self.pcfArray.reshape((self.nBinsT, self.nBinsY, self.nBinsX))
        self.avgOccupancy = numpy.sum(numpy.sum(numpy.sum(self.pcfArray))) / (self.nBinsX * self.nBinsY * self.nBinsT)
        self.pmfArray = -numpy.log(numpy.copy(self.pcfArray))

    ## Reset the PCF array
    # Call when you want to zero out the pcf array in C
    # This should be done if you need to change the types of particles being compared
    def reset(self):
        self.pmfHandle.resetPMF()

## Computes the 2D anisotropic potential of mean force and torque using \theta_+ = \phi_1 + \phi_2, \theta_- = \phi_1 - \phi_2
class pmftRPM(object):
    ## Initialize the pmftXYT2D object:
    # \param box The simulation box from freud
    # \param maxR The maximum distance to consider
    # \param maxTP The maximum angle sum to consider
    # \param maxTM The maximum angle difference
    # \param dr The distance bin size
    # \param dTP The angle sum bin size
    # \param dTM The angle difference bin size
    def __init__(self, maxR, maxTP, maxTM, dr, dTP, dTM):
        super(pmftRPM, self).__init__()
        self.box = box
        self.maxR = maxR
        self.maxTP = maxTP
        self.maxTM = maxTM
        self.dr = dr
        self.dTP = dTP
        self.dTM = dTM
        self.pmftHandle = PMFTRPM(self.maxR, self.maxTP, self.maxTM, self.dr, self.dTP, self.dTM)
        self.rArray = self.pmfHandle.getR()
        self.TPArray = self.pmfHandle.getTP()
        self.TMArray = self.pmfHandle.getTM()
        self.nBinsR = int(len(self.rArray))
        self.nBinsTP = int(len(self.TPArray))
        self.nBinsTM = int(len(self.TMArray))

    ## Compute the pmft for a given set of points (one traj frame)
    # \param refPos Reference point to consider
    # \param refAng Reference angles to consider as floats
    # \param pos Points to consider
    # \param ang Angles to consider as floats
    def compute(self, box, refPos=None, refAng=None, pos=None, ang=None):
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
        self.pmftHandle.compute(box, self.refPos, self.refAng, self.pos, self.ang)

    ## Calculate the PMF from the PCF. This has the side-effect of also populating the self.pcfArray
    # in addition to self.pmfArray
    # after calling calcPMF(), you can access the results via:
    # self.pcfArray for the positional correlation function
    # self.avgOccupancy for the average bin occupancy (useful for sanity checking)
    # self.pmfArray will give the pmf
    # NOTE: self.pcfArray is of type numpy.uint32; if averaging multiple frames,
    # you need to recast as floats: self.pcfArray.astype(numpy.float32)
    def calcPMF(self):
        self.pcfArray = self.pmfHandle.getPCF()
        self.pcfArray = self.pcfArray.reshape((self.nBinsTM, self.nBinsTP, self.nBinsR))
        self.avgOccupancy = numpy.sum(numpy.sum(numpy.sum(self.pcfArray))) / (self.nBinsR * self.nBinsTP * self.nBinsTM)
        self.pmfArray = -numpy.log(numpy.copy(self.pcfArray))

    ## Reset the PCF array
    # Call when you want to zero out the pcf array in C
    # This should be done if you need to change the types of particles being compared
    def reset(self):
        self.pmfHandle.resetPMF()
