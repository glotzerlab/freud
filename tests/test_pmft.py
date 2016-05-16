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

        myPMFT = pmft.PMFXYZ(maxX, maxY, maxZ, nbinsX, nbinsY, nbinsZ)

        # get the info from pmft

        xArray = numpy.copy(myPMFT.getX())
        yArray = numpy.copy(myPMFT.getY())
        zArray = numpy.copy(myPMFT.getZ())

        npt.assert_almost_equal(xArray, listX, decimal=3)
        npt.assert_almost_equal(yArray, listY, decimal=3)
        npt.assert_almost_equal(zArray, listZ, decimal=3)

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
        myPMFT = pmft.PMFXY2D(maxX, maxY, nbinsX, nbinsY)
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
        myPMFT = pmft.PMFXY2D(maxX, maxY, nbinsX, nbinsY)
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
        myPMFT = pmft.PMFXY2D(maxX, maxY, nbinsX, nbinsY)
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
        myPMFT = pmft.PMFXY2D(maxX, maxY, nbinsX, nbinsY)
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
