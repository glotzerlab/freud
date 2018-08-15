import numpy as np
import numpy.testing as npt
from freud import box, pmft
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

        myPMFT = pmft.PMFTXYZ(maxX, maxY, maxZ, nbinsX, nbinsY, nbinsZ)

        # get the info from pmft

        xArray = np.copy(myPMFT.X)
        yArray = np.copy(myPMFT.Y)
        zArray = np.copy(myPMFT.Z)

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
        dT1 = (2.0 * np.pi / float(nbinsT1))
        dT2 = (2.0 * np.pi / float(nbinsT2))

        # make sure the radius for each bin is generated correctly
        listR = np.zeros(nbinsR, dtype=np.float32)
        listT1 = np.zeros(nbinsT1, dtype=np.float32)
        listT2 = np.zeros(nbinsT2, dtype=np.float32)

        for i in range(nbinsR):
            r = float(i) * dr
            nextr = float(i + 1) * dr
            listR[i] = 2.0/3.0 * (
                nextr*nextr*nextr - r*r*r)/(nextr*nextr - r*r)

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

        rArray = np.copy(myPMFT.R)
        T1Array = np.copy(myPMFT.T1)
        T2Array = np.copy(myPMFT.T2)

        npt.assert_almost_equal(rArray, listR, decimal=3)
        npt.assert_almost_equal(T1Array, listT1, decimal=3)
        npt.assert_almost_equal(T2Array, listT2, decimal=3)

        npt.assert_equal(nbinsR, myPMFT.n_bins_r)
        npt.assert_equal(nbinsT1, myPMFT.n_bins_T1)
        npt.assert_equal(nbinsT2, myPMFT.n_bins_T2)

        pmftArr = myPMFT.PCF
        npt.assert_equal(nbinsR, pmftArr.shape[0])
        npt.assert_equal(nbinsT1, pmftArr.shape[2])
        npt.assert_equal(nbinsT2, pmftArr.shape[1])


class TestPMFXY2DAccumulate(unittest.TestCase):
    def test_twoParticlesWithCellList(self):
        boxSize = 16.0
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                          dtype=np.float32)
        angles = np.array([0.0, 0.0], dtype=np.float32)
        maxX = 3.0
        maxY = 3.0
        nbinsX = 100
        nbinsY = 100
        dx = (2.0 * maxX / float(nbinsX))
        dy = (2.0 * maxY / float(nbinsY))
        myPMFT = pmft.PMFTXY2D(maxX, maxY, nbinsX, nbinsY)
        myPMFT.accumulate(box.Box.square(boxSize), points, angles,
                          points, angles)

        correct = np.zeros(shape=(len(myPMFT.Y), len(myPMFT.X)),
                           dtype=np.int32)
        # calculation for array idxs
        # particle 0
        deltaX = points[0][0] - points[1][0]
        deltaY = points[0][1] - points[1][1]
        x = deltaX + maxX
        y = deltaY + maxY
        binX = int(np.floor(x / dx))
        binY = int(np.floor(y / dy))
        correct[binY, binX] = 1
        deltaX = points[1][0] - points[0][0]
        deltaY = points[1][1] - points[0][1]
        x = deltaX + maxX
        y = deltaY + maxY
        binX = int(np.floor(x / dx))
        binY = int(np.floor(y / dy))
        correct[binY, binX] = 1
        absoluteTolerance = 0.1
        pcfArray = myPMFT.bin_counts
        npt.assert_allclose(pcfArray, correct, atol=absoluteTolerance)

    def test_twoParticlesWithoutCellList(self):
        boxSize = 16.0
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                          dtype=np.float32)
        angles = np.array([0.0, 0.0], dtype=np.float32)
        maxX = 3.0
        maxY = 3.0
        nbinsX = 100
        nbinsY = 100
        dx = (2.0 * maxX / float(nbinsX))
        dy = (2.0 * maxY / float(nbinsY))
        myPMFT = pmft.PMFTXY2D(maxX, maxY, nbinsX, nbinsY)
        myPMFT.accumulate(box.Box.square(boxSize), points, angles,
                          points, angles)

        correct = np.zeros(shape=(len(myPMFT.Y), len(myPMFT.X)),
                           dtype=np.int32)
        # calculation for array idxs
        # particle 0
        deltaX = points[0][0] - points[1][0]
        deltaY = points[0][1] - points[1][1]
        x = deltaX + maxX
        y = deltaY + maxY
        binX = int(np.floor(x / dx))
        binY = int(np.floor(y / dy))
        correct[binY, binX] = 1
        deltaX = points[1][0] - points[0][0]
        deltaY = points[1][1] - points[0][1]
        x = deltaX + maxX
        y = deltaY + maxY
        binX = int(np.floor(x / dx))
        binY = int(np.floor(y / dy))
        correct[binY, binX] = 1
        absoluteTolerance = 0.1
        pcfArray = myPMFT.bin_counts
        npt.assert_allclose(pcfArray, correct, atol=absoluteTolerance)


class TestPMFXY2DCompute(unittest.TestCase):
    def test_twoParticlesWithCellList(self):
        boxSize = 16.0
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                          dtype=np.float32)
        angles = np.array([0.0, 0.0], dtype=np.float32)
        maxX = 3.0
        maxY = 3.0
        dx = 0.1
        dy = 0.1
        nbinsX = int(2 * int(np.floor(maxX / dx)))
        nbinsY = int(2 * int(np.floor(maxY / dy)))
        myPMFT = pmft.PMFTXY2D(maxX, maxY, nbinsX, nbinsY)
        myPMFT.compute(box.Box.square(boxSize), points, angles,
                       points, angles)

        correct = np.zeros(shape=(len(myPMFT.Y), len(myPMFT.X)),
                           dtype=np.int32)
        # calculation for array idxs
        # particle 0
        deltaX = points[0][0] - points[1][0]
        deltaY = points[0][1] - points[1][1]
        x = deltaX + maxX
        y = deltaY + maxY
        binX = int(np.floor(x / dx))
        binY = int(np.floor(y / dy))
        correct[binY, binX] = 1
        deltaX = points[1][0] - points[0][0]
        deltaY = points[1][1] - points[0][1]
        x = deltaX + maxX
        y = deltaY + maxY
        binX = int(np.floor(x / dx))
        binY = int(np.floor(y / dy))
        correct[binY, binX] = 1
        absoluteTolerance = 0.1
        pcfArray = myPMFT.bin_counts
        npt.assert_allclose(pcfArray, correct, atol=absoluteTolerance)

    def test_twoParticlesWithoutCellList(self):
        boxSize = 16.0
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                          dtype=np.float32)
        angles = np.array([0.0, 0.0], dtype=np.float32)
        maxX = 3.0
        maxY = 3.0
        dx = 0.1
        dy = 0.1
        nbinsX = int(2 * int(np.floor(maxX / dx)))
        nbinsY = int(2 * int(np.floor(maxY / dy)))
        myPMFT = pmft.PMFTXY2D(maxX, maxY, nbinsX, nbinsY)
        myPMFT.compute(box.Box.square(boxSize), points, angles, points, angles)

        correct = np.zeros(shape=(len(myPMFT.Y), len(myPMFT.X)),
                           dtype=np.int32)
        # calculation for array idxs
        # particle 0
        deltaX = points[0][0] - points[1][0]
        deltaY = points[0][1] - points[1][1]
        x = deltaX + maxX
        y = deltaY + maxY
        binX = int(np.floor(x / dx))
        binY = int(np.floor(y / dy))
        correct[binY, binX] = 1
        deltaX = points[1][0] - points[0][0]
        deltaY = points[1][1] - points[0][1]
        x = deltaX + maxX
        y = deltaY + maxY
        binX = int(np.floor(x / dx))
        binY = int(np.floor(y / dy))
        correct[binY, binX] = 1
        absoluteTolerance = 0.1
        pcfArray = myPMFT.bin_counts
        npt.assert_allclose(pcfArray, correct, atol=absoluteTolerance)


class TestPMFTXYZShift(unittest.TestCase):
    def test_two_particles_dead_pixel(self):
        points = np.array([[1, 1, 1], [0, 0, 0]], dtype=np.float32)
        orientations = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
        noshift = pmft.PMFTXYZ(0.5, 0.5, 0.5, 3, 3, 3, shiftvec=[0, 0, 0])
        shift = pmft.PMFTXYZ(0.5, 0.5, 0.5, 3, 3, 3, shiftvec=[1, 1, 1])

        for pm in [noshift, shift]:
            pm.compute(box.Box.cube(3), points, orientations,
                       points, orientations, face_orientations=None)

        # Non-shifted pmft should have no non-inf valued voxels,
        # since the other point is outside the x/y/z max
        infcheck_noshift = np.logical_not(np.isinf(noshift.PMFT)).sum()
        # Shifted pmft should have one non-inf valued voxel
        infcheck_shift = np.logical_not(np.isinf(shift.PMFT)).sum()

        assert(infcheck_noshift == 0)
        assert(infcheck_shift == 1)


if __name__ == '__main__':
    unittest.main()
