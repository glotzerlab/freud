import numpy as np
import numpy.testing as npt
from freud import trajectory, pmft
import unittest

class TestBins(unittest.TestCase):
    # this just tests from pmftXYZ but might want to add in others...
    def test_generateBins(self):
        maxX = 51.23
        dx = 0.1
        maxY = 51.23
        dy = 0.1
        maxZ = 51.23
        dz = 0.1
        nbinsX = int(2 * floorf(maxX / dx))
        nbinsY = int(2 * floorf(maxX / dy))
        nbinsZ = int(2 * floorf(maxX / dz))

        # make sure the radius for each bin is generated correctly
        listX = np.zeros(nbinsX, dtype=np.float32)
        listY = np.zeros(nbinsY, dtype=np.float32)
        listZ = np.zeros(nbinsZ, dtype=np.float32)

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

        myPMFT = pmft.pmftXYZ(box=trajectory.Box(maxX*3.1, maxY*3.1, maxZ*3.1, False),
                              maxX=maxX,
                              maxY=maxY,
                              maxZ=maxZ,
                              dx=dx,
                              dy=dy,
                              dz=dz)

        # get the info from pmft

        xArray = np.copy(myPMFT.pmftHandle.getX())
        yArray = np.copy(myPMFT.pmftHandle.getY())
        zArray = np.copy(myPMFT.pmftHandle.getZ())

        npt.assert_almost_equal(xArray, listX, decimal=3)
        npt.assert_almost_equal(yArray, listY, decimal=3)
        npt.assert_almost_equal(zArray, listZ, decimal=3)

class TestRDF(unittest.TestCase):
    def test_twoParticlesWithCellList(self):
        boxSize = 16.0
        points = numpy.array([[-1.0, 0.0], [1.0, 0.0]], dtype=numpy.float32)
        angles = numpy.array([0.0, 0.0], dtype=numpy.float32)
        maxX = 3.0
        maxY = 3.0
        dx = 0.1
        dy = 0.1
        myPMFT = pmft.pmfXY2D(box=trajectory.Box(boxSize, boxSize, 0, True),
                              maxX=maxX,
                              maxY=maxY,
                              dx=dx,
                              dy=dy)
        myPMFT.compute(points, angles, points, angles)

        correct = np.zeros(shape=(myPMFT.nBinsY, myPMFT.nBinsX), dtype=np.float32)
        # calculation for array idxs
        # particle 0
        deltaX = points[0][0] - points[1][0]
        deltaY = points[0][1] - points[1][1]
        x = deltaX + maxX
        y = deltaY + maxY
        binX = numpy.floor(x / dx)
        binY = numpy.floor(y / dy)
        correct[binY][binX] = 1
        deltaX = points[1][0] - points[0][0]
        deltaY = points[1][1] - points[0][1]
        x = deltaX + maxX
        y = deltaY + maxY
        binX = numpy.floor(x / dx)
        binY = numpy.floor(y / dy)
        correct[binY][binX] = 1
        absoluteTolerance = 0.1
        npt.assert_allclose(myPMFT.pcfArray, correct, atol=absoluteTolerance)

    def test_random_point_without_cell_list(self):
        boxSize = 16.0
        points = numpy.array([[-1.0, 0.0], [1.0, 0.0]], dtype=numpy.float32)
        angles = numpy.array([0.0, 0.0], dtype=numpy.float32)
        maxX = 3.0
        maxY = 3.0
        dx = 0.1
        dy = 0.1
        myPMFT = pmft.pmfXY2D(box=trajectory.Box(boxSize, boxSize, 0, True),
                              maxX=maxX,
                              maxY=maxY,
                              dx=dx,
                              dy=dy)
        myPMFT.compute(points, angles, points, angles)

        correct = np.zeros(shape=(myPMFT.nBinsY, myPMFT.nBinsX), dtype=np.float32)
        # calculation for array idxs
        # particle 0
        deltaX = points[0][0] - points[1][0]
        deltaY = points[0][1] - points[1][1]
        x = deltaX + maxX
        y = deltaY + maxY
        binX = numpy.floor(x / dx)
        binY = numpy.floor(y / dy)
        correct[binY][binX] = 1
        deltaX = points[1][0] - points[0][0]
        deltaY = points[1][1] - points[0][1]
        x = deltaX + maxX
        y = deltaY + maxY
        binX = numpy.floor(x / dx)
        binY = numpy.floor(y / dy)
        correct[binY][binX] = 1
        absoluteTolerance = 0.1
        npt.assert_allclose(myPMFT.pcfArray, correct, atol=absoluteTolerance)



if __name__ == '__main__':
    unittest.main()
