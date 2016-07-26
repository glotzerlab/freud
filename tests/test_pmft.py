import numpy
import numpy.testing as npt
from freud import trajectory, pmft
import unittest

class TestBins(unittest.TestCase):
    # this just tests from pmftXYZ but might want to add in others...
    def test_generateBins(self):
        maxX = 5.23
        nbinsX = 100
        maxY = 5.23
        nbinsY = 100
        maxZ = 5.23
        nbinsZ = 100
        dx = (2.0 * maxX / float(nbinsX))
        dy = (2.0 * maxY / float(nbinsY))
        dz = (2.0 * maxZ / float(nbinsZ))

        # make sure the radius for each bin is generated correctly
        listX = numpy.zeros(nbinsX, dtype=numpy.float32)
        listY = numpy.zeros(nbinsY, dtype=numpy.float32)
        listZ = numpy.zeros(nbinsZ, dtype=numpy.float32)

        for i in range(nbinsX):
            x = float(i) * dx
            nextX = float(i + 1) * dx
            listX[i] = -maxX + ((x + nextX) / 2.0)

        for i in range(nbinsY):
            y = float(i) * dy
            nextY = float(i + 1) * dy
            listY[i] = -maxY + ((y + nextY) / 2.0)

        for i in range(nbinsZ):
            z = float(i) * dz
            nextZ = float(i + 1) * dz
            listZ[i] = -maxZ + ((z + nextZ) / 2.0)

        myPMFT = pmft.PMFTXYZ(maxX, maxY, maxZ, nbinsX, nbinsY, nbinsZ)

        # get the info from pmft

        xArray = numpy.copy(myPMFT.getX())
        yArray = numpy.copy(myPMFT.getY())
        zArray = numpy.copy(myPMFT.getZ())

        npt.assert_almost_equal(xArray, listX, decimal=3)
        npt.assert_almost_equal(yArray, listY, decimal=3)
        npt.assert_almost_equal(zArray, listZ, decimal=3)

class TestBinsR12(unittest.TestCase):
    # this just tests from pmftXYZ but might want to add in others...
    def test_generateBins(self):
        maxR = 5.23
        nbinsR = 10
        nbinsT1 = 20
        nbinsT2 = 30
        dr = (maxR / float(nbinsR))
        dT1 = (2.0 * numpy.pi / float(nbinsT1))
        dT2 = (2.0 * numpy.pi / float(nbinsT2))

        # make sure the radius for each bin is generated correctly
        listR = numpy.zeros(nbinsR, dtype=numpy.float32)
        listT1 = numpy.zeros(nbinsT1, dtype=numpy.float32)
        listT2 = numpy.zeros(nbinsT2, dtype=numpy.float32)

        for i in range(nbinsR):
            r = float(i) * dr
            nextr = float(i + 1) * dr
            listR[i] = 2.0/3.0 * (nextr*nextr*nextr - r*r*r)/(nextr*nextr - r*r)

        for i in range(nbinsT1):
            t = float(i) * dT1
            nextt = float(i + 1) * dT1
            listT1[i] = ((t + nextt) / 2.0)

        for i in range(nbinsT2):
            t = float(i) * dT2
            nextt = float(i + 1) * dT2
            listT2[i] = ((t + nextt) / 2.0)

        myPMFT = pmft.PMFTR12(maxR, nbinsR, nbinsT1, nbinsT2)

        # get the info from pmft

        rArray = numpy.copy(myPMFT.getR())
        T1Array = numpy.copy(myPMFT.getT1())
        T2Array = numpy.copy(myPMFT.getT2())

        npt.assert_almost_equal(rArray, listR, decimal=3)
        npt.assert_almost_equal(T1Array, listT1, decimal=3)
        npt.assert_almost_equal(T2Array, listT2, decimal=3)

        npt.assert_equal(nbinsR, myPMFT.getNBinsR())
        npt.assert_equal(nbinsT1, myPMFT.getNBinsT1())
        npt.assert_equal(nbinsT2, myPMFT.getNBinsT2())

        pmftArr = myPMFT.getPCF()
        npt.assert_equal(nbinsR, pmftArr.shape[0])
        npt.assert_equal(nbinsT1, pmftArr.shape[2])
        npt.assert_equal(nbinsT2, pmftArr.shape[1])

class TestPMFXY2DAccumulate(unittest.TestCase):
    def test_twoParticlesWithCellList(self):
        boxSize = 16.0
        points = numpy.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=numpy.float32)
        angles = numpy.array([0.0, 0.0], dtype=numpy.float32)
        maxX = 3.0
        maxY = 3.0
        nbinsX = 100
        nbinsY = 100
        dx = (2.0 * maxX / float(nbinsX))
        dy = (2.0 * maxY / float(nbinsY))
        myPMFT = pmft.PMFTXY2D(maxX, maxY, nbinsX, nbinsY)
        myPMFT.accumulate(trajectory.Box(boxSize, boxSize, 0, True), points, angles, points, angles)

        correct = numpy.zeros(shape=(len(myPMFT.getY()), len(myPMFT.getX())), dtype=numpy.int32)
        # calculation for array idxs
        # particle 0
        deltaX = points[0][0] - points[1][0]
        deltaY = points[0][1] - points[1][1]
        x = deltaX + maxX
        y = deltaY + maxY
        binX = numpy.floor(x / dx)
        binY = numpy.floor(y / dy)
        correct[binY,binX] = 1
        deltaX = points[1][0] - points[0][0]
        deltaY = points[1][1] - points[0][1]
        x = deltaX + maxX
        y = deltaY + maxY
        binX = numpy.floor(x / dx)
        binY = numpy.floor(y / dy)
        correct[binY,binX] = 1
        absoluteTolerance = 0.1
        pcfArray = myPMFT.getBinCounts()
        npt.assert_allclose(pcfArray, correct, atol=absoluteTolerance)

    def test_twoParticlesWithoutCellList(self):
        boxSize = 16.0
        points = numpy.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=numpy.float32)
        angles = numpy.array([0.0, 0.0], dtype=numpy.float32)
        maxX = 3.0
        maxY = 3.0
        nbinsX = 100
        nbinsY = 100
        dx = (2.0 * maxX / float(nbinsX))
        dy = (2.0 * maxY / float(nbinsY))
        myPMFT = pmft.PMFTXY2D(maxX, maxY, nbinsX, nbinsY)
        myPMFT.accumulate(trajectory.Box(boxSize, boxSize, 0, True), points, angles, points, angles)

        correct = numpy.zeros(shape=(len(myPMFT.getY()), len(myPMFT.getX())), dtype=numpy.int32)
        # calculation for array idxs
        # particle 0
        deltaX = points[0][0] - points[1][0]
        deltaY = points[0][1] - points[1][1]
        x = deltaX + maxX
        y = deltaY + maxY
        binX = numpy.floor(x / dx)
        binY = numpy.floor(y / dy)
        correct[binY,binX] = 1
        deltaX = points[1][0] - points[0][0]
        deltaY = points[1][1] - points[0][1]
        x = deltaX + maxX
        y = deltaY + maxY
        binX = numpy.floor(x / dx)
        binY = numpy.floor(y / dy)
        correct[binY,binX] = 1
        absoluteTolerance = 0.1
        pcfArray = myPMFT.getBinCounts()
        npt.assert_allclose(pcfArray, correct, atol=absoluteTolerance)

class TestPMFXY2DCompute(unittest.TestCase):
    def test_twoParticlesWithCellList(self):
        boxSize = 16.0
        points = numpy.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=numpy.float32)
        angles = numpy.array([0.0, 0.0], dtype=numpy.float32)
        maxX = 3.0
        maxY = 3.0
        dx = 0.1
        dy = 0.1
        nbinsX = int(2 * numpy.floor(maxX / dx))
        nbinsY = int(2 * numpy.floor(maxY / dy))
        myPMFT = pmft.PMFTXY2D(maxX, maxY, nbinsX, nbinsY)
        myPMFT.compute(trajectory.Box(boxSize, boxSize, 0, True), points, angles, points, angles)

        correct = numpy.zeros(shape=(len(myPMFT.getY()), len(myPMFT.getX())), dtype=numpy.int32)
        # calculation for array idxs
        # particle 0
        deltaX = points[0][0] - points[1][0]
        deltaY = points[0][1] - points[1][1]
        x = deltaX + maxX
        y = deltaY + maxY
        binX = numpy.floor(x / dx)
        binY = numpy.floor(y / dy)
        correct[binY,binX] = 1
        deltaX = points[1][0] - points[0][0]
        deltaY = points[1][1] - points[0][1]
        x = deltaX + maxX
        y = deltaY + maxY
        binX = numpy.floor(x / dx)
        binY = numpy.floor(y / dy)
        correct[binY,binX] = 1
        absoluteTolerance = 0.1
        pcfArray = myPMFT.getBinCounts()
        npt.assert_allclose(pcfArray, correct, atol=absoluteTolerance)

    def test_twoParticlesWithoutCellList(self):
        boxSize = 16.0
        points = numpy.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=numpy.float32)
        angles = numpy.array([0.0, 0.0], dtype=numpy.float32)
        maxX = 3.0
        maxY = 3.0
        dx = 0.1
        dy = 0.1
        nbinsX = int(2 * numpy.floor(maxX / dx))
        nbinsY = int(2 * numpy.floor(maxY / dy))
        myPMFT = pmft.PMFTXY2D(maxX, maxY, nbinsX, nbinsY)
        myPMFT.compute(trajectory.Box(boxSize, boxSize, 0, True), points, angles, points, angles)

        correct = numpy.zeros(shape=(len(myPMFT.getY()), len(myPMFT.getX())), dtype=numpy.int32)
        # calculation for array idxs
        # particle 0
        deltaX = points[0][0] - points[1][0]
        deltaY = points[0][1] - points[1][1]
        x = deltaX + maxX
        y = deltaY + maxY
        binX = numpy.floor(x / dx)
        binY = numpy.floor(y / dy)
        correct[binY,binX] = 1
        deltaX = points[1][0] - points[0][0]
        deltaY = points[1][1] - points[0][1]
        x = deltaX + maxX
        y = deltaY + maxY
        binX = numpy.floor(x / dx)
        binY = numpy.floor(y / dy)
        correct[binY,binX] = 1
        absoluteTolerance = 0.1
        pcfArray = myPMFT.getBinCounts()
        npt.assert_allclose(pcfArray, correct, atol=absoluteTolerance)

if __name__ == '__main__':
    print("testing pmft")
    unittest.main()
