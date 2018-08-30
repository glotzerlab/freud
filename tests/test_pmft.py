import numpy as np
import numpy.testing as npt
import freud
from freud.errors import FreudDeprecationWarning
import unittest
import warnings


class TestPMFTR12(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=FreudDeprecationWarning)

    def test_box(self):
        boxSize = 16.0
        box = freud.box.Box.square(boxSize)
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                          dtype=np.float32)
        angles = np.array([0.0, 0.0], dtype=np.float32)
        maxR = 5.23
        nbinsR = 10
        nbinsT1 = 20
        nbinsT2 = 30
        myPMFT = freud.pmft.PMFTR12(maxR, nbinsR, nbinsT1, nbinsT2)
        myPMFT.accumulate(box, points, angles, points, angles)
        npt.assert_equal(myPMFT.box, freud.box.Box.square(boxSize))
        # Test old methods
        npt.assert_equal(myPMFT.getBox(), freud.box.Box.square(boxSize))

    def test_bins(self):
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

        myPMFT = freud.pmft.PMFTR12(maxR, nbinsR, nbinsT1, nbinsT2)

        # Compare expected bins to the info from pmft
        npt.assert_almost_equal(myPMFT.R, listR, decimal=3)
        npt.assert_almost_equal(myPMFT.T1, listT1, decimal=3)
        npt.assert_almost_equal(myPMFT.T2, listT2, decimal=3)

        npt.assert_equal(nbinsR, myPMFT.n_bins_R)
        npt.assert_equal(nbinsT1, myPMFT.n_bins_T1)
        npt.assert_equal(nbinsT2, myPMFT.n_bins_T2)

        # Test old methods
        npt.assert_equal(nbinsR, myPMFT.getNBinsR())
        npt.assert_equal(nbinsT1, myPMFT.getNBinsT1())
        npt.assert_equal(nbinsT2, myPMFT.getNBinsT2())

        npt.assert_equal(myPMFT.PCF.shape, (nbinsR, nbinsT2, nbinsT1))
        npt.assert_equal(myPMFT.PMFT.shape, (nbinsR, nbinsT2, nbinsT1))

        # Test old methods
        npt.assert_equal(myPMFT.getPCF().shape, (nbinsR, nbinsT2, nbinsT1))
        npt.assert_equal(myPMFT.getPMFT().shape, (nbinsR, nbinsT2, nbinsT1))


class TestPMFTXYT(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=FreudDeprecationWarning)

    def test_box(self):
        boxSize = 16.0
        box = freud.box.Box.square(boxSize)
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                          dtype=np.float32)
        angles = np.array([0.0, 0.0], dtype=np.float32)
        maxX = 3.0
        maxY = 4.0
        nbinsX = 20
        nbinsY = 30
        nbinsT = 40
        myPMFT = freud.pmft.PMFTXYT(maxX, maxY, nbinsX, nbinsY, nbinsT)
        myPMFT.accumulate(box, points, angles, points, angles)
        npt.assert_equal(myPMFT.box, freud.box.Box.square(boxSize))
        # Test old methods
        npt.assert_equal(myPMFT.getBox(), freud.box.Box.square(boxSize))

    def test_bins(self):
        maxX = 3.0
        maxY = 4.0
        nbinsX = 20
        nbinsY = 30
        nbinsT = 40
        dx = (2.0 * maxX / float(nbinsX))
        dy = (2.0 * maxY / float(nbinsY))
        dT = (2.0 * np.pi / float(nbinsT))

        # make sure the center for each bin is generated correctly
        listX = np.zeros(nbinsX, dtype=np.float32)
        listY = np.zeros(nbinsY, dtype=np.float32)
        listT = np.zeros(nbinsT, dtype=np.float32)

        for i in range(nbinsX):
            x = float(i) * dx
            nextX = float(i + 1) * dx
            listX[i] = -maxX + ((x + nextX) / 2.0)

        for i in range(nbinsY):
            y = float(i) * dy
            nextY = float(i + 1) * dy
            listY[i] = -maxY + ((y + nextY) / 2.0)

        for i in range(nbinsT):
            t = float(i) * dT
            nextt = float(i + 1) * dT
            listT[i] = ((t + nextt) / 2.0)

        myPMFT = freud.pmft.PMFTXYT(maxX, maxY, nbinsX, nbinsY, nbinsT)

        # Compare expected bins to the info from pmft
        npt.assert_almost_equal(myPMFT.X, listX, decimal=3)
        npt.assert_almost_equal(myPMFT.Y, listY, decimal=3)
        npt.assert_almost_equal(myPMFT.T, listT, decimal=3)

        npt.assert_equal(nbinsX, myPMFT.n_bins_X)
        npt.assert_equal(nbinsY, myPMFT.n_bins_Y)
        npt.assert_equal(nbinsT, myPMFT.n_bins_T)

        # Test old methods
        npt.assert_equal(nbinsX, myPMFT.getNBinsX())
        npt.assert_equal(nbinsY, myPMFT.getNBinsY())
        npt.assert_equal(nbinsT, myPMFT.getNBinsT())

        npt.assert_equal(myPMFT.PCF.shape, (nbinsT, nbinsY, nbinsX))
        npt.assert_equal(myPMFT.PMFT.shape, (nbinsT, nbinsY, nbinsX))

        # Test old methods
        npt.assert_equal(myPMFT.getPCF().shape, (nbinsT, nbinsY, nbinsX))
        npt.assert_equal(myPMFT.getPMFT().shape, (nbinsT, nbinsY, nbinsX))


class TestPMFTXY2D(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=FreudDeprecationWarning)

    def test_box(self):
        boxSize = 16.0
        box = freud.box.Box.square(boxSize)
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                          dtype=np.float32)
        angles = np.array([0.0, 0.0], dtype=np.float32)
        maxX = 3.0
        maxY = 4.0
        nbinsX = 100
        nbinsY = 110
        myPMFT = freud.pmft.PMFTXY2D(maxX, maxY, nbinsX, nbinsY)
        myPMFT.accumulate(box, points, angles, points, angles)
        npt.assert_equal(myPMFT.box, freud.box.Box.square(boxSize))
        # Test old methods
        npt.assert_equal(myPMFT.getBox(), freud.box.Box.square(boxSize))

    def test_bins(self):
        maxX = 3.0
        maxY = 4.0
        nbinsX = 20
        nbinsY = 30
        dx = (2.0 * maxX / float(nbinsX))
        dy = (2.0 * maxY / float(nbinsY))

        # make sure the center for each bin is generated correctly
        listX = np.zeros(nbinsX, dtype=np.float32)
        listY = np.zeros(nbinsY, dtype=np.float32)

        for i in range(nbinsX):
            x = float(i) * dx
            nextX = float(i + 1) * dx
            listX[i] = -maxX + ((x + nextX) / 2.0)

        for i in range(nbinsY):
            y = float(i) * dy
            nextY = float(i + 1) * dy
            listY[i] = -maxY + ((y + nextY) / 2.0)

        myPMFT = freud.pmft.PMFTXY2D(maxX, maxY, nbinsX, nbinsY)

        # Compare expected bins to the info from pmft
        npt.assert_almost_equal(myPMFT.X, listX, decimal=3)
        npt.assert_almost_equal(myPMFT.Y, listY, decimal=3)

        npt.assert_equal(nbinsX, myPMFT.n_bins_X)
        npt.assert_equal(nbinsY, myPMFT.n_bins_Y)

        # Test old methods
        npt.assert_equal(nbinsX, myPMFT.getNBinsX())
        npt.assert_equal(nbinsY, myPMFT.getNBinsY())

        npt.assert_equal(myPMFT.PCF.shape, (nbinsY, nbinsX))
        npt.assert_equal(myPMFT.PMFT.shape, (nbinsY, nbinsX))

        # Test old methods
        npt.assert_equal(myPMFT.getPCF().shape, (nbinsY, nbinsX))
        npt.assert_equal(myPMFT.getPMFT().shape, (nbinsY, nbinsX))

    def test_two_particles(self):
        boxSize = 16.0
        box = freud.box.Box.square(boxSize)
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                          dtype=np.float32)
        angles = np.array([0.0, 0.0], dtype=np.float32)
        maxX = 3.0
        maxY = 4.0
        nbinsX = 100
        nbinsY = 110
        dx = (2.0 * maxX / float(nbinsX))
        dy = (2.0 * maxY / float(nbinsY))

        correct_bin_counts = np.zeros(shape=(nbinsY, nbinsX), dtype=np.int32)
        # calculation for array idxs
        # particle 0
        deltaX = points[0][0] - points[1][0]
        deltaY = points[0][1] - points[1][1]
        x = deltaX + maxX
        y = deltaY + maxY
        binX = int(np.floor(x / dx))
        binY = int(np.floor(y / dy))
        correct_bin_counts[binY, binX] = 1
        deltaX = points[1][0] - points[0][0]
        deltaY = points[1][1] - points[0][1]
        x = deltaX + maxX
        y = deltaY + maxY
        binX = int(np.floor(x / dx))
        binY = int(np.floor(y / dy))
        correct_bin_counts[binY, binX] = 1
        absoluteTolerance = 0.1

        myPMFT = freud.pmft.PMFTXY2D(maxX, maxY, nbinsX, nbinsY)
        myPMFT.accumulate(box, points, angles, points, angles)
        npt.assert_allclose(myPMFT.bin_counts, correct_bin_counts,
                            atol=absoluteTolerance)
        myPMFT.compute(box, points, angles, points, angles)
        npt.assert_allclose(myPMFT.bin_counts, correct_bin_counts,
                            atol=absoluteTolerance)
        myPMFT.reset()
        npt.assert_allclose(myPMFT.bin_counts, 0,
                            atol=absoluteTolerance)


class TestPMFTXYZ(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=FreudDeprecationWarning)

    def test_box(self):
        boxSize = 25.0
        box = freud.box.Box.cube(boxSize)
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                          dtype=np.float32)
        orientations = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
        maxX = 5.23
        maxY = 6.23
        maxZ = 7.23
        nbinsX = 100
        nbinsY = 110
        nbinsZ = 120
        myPMFT = freud.pmft.PMFTXYZ(maxX, maxY, maxZ, nbinsX, nbinsY, nbinsZ)
        myPMFT.accumulate(box, points, orientations, points, orientations)
        npt.assert_equal(myPMFT.box, freud.box.Box.cube(boxSize))
        # Test old methods
        npt.assert_equal(myPMFT.getBox(), freud.box.Box.cube(boxSize))

    def test_bins(self):
        maxX = 5.23
        maxY = 6.23
        maxZ = 7.23
        nbinsX = 100
        nbinsY = 110
        nbinsZ = 120
        dx = (2.0 * maxX / float(nbinsX))
        dy = (2.0 * maxY / float(nbinsY))
        dz = (2.0 * maxZ / float(nbinsZ))

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

        myPMFT = freud.pmft.PMFTXYZ(maxX, maxY, maxZ, nbinsX, nbinsY, nbinsZ)

        # Compare expected bins to the info from pmft
        npt.assert_almost_equal(myPMFT.X, listX, decimal=3)
        npt.assert_almost_equal(myPMFT.Y, listY, decimal=3)
        npt.assert_almost_equal(myPMFT.Z, listZ, decimal=3)

        npt.assert_equal(nbinsX, myPMFT.n_bins_X)
        npt.assert_equal(nbinsY, myPMFT.n_bins_Y)
        npt.assert_equal(nbinsZ, myPMFT.n_bins_Z)

        # Test old methods
        npt.assert_equal(nbinsX, myPMFT.getNBinsX())
        npt.assert_equal(nbinsY, myPMFT.getNBinsY())
        npt.assert_equal(nbinsZ, myPMFT.getNBinsZ())

        npt.assert_equal(myPMFT.PCF.shape, (nbinsZ, nbinsY, nbinsX))
        npt.assert_equal(myPMFT.PMFT.shape, (nbinsZ, nbinsY, nbinsX))

        # Test old methods
        npt.assert_equal(myPMFT.getPCF().shape, (nbinsZ, nbinsY, nbinsX))
        npt.assert_equal(myPMFT.getPMFT().shape, (nbinsZ, nbinsY, nbinsX))

    def test_shift_two_particles_dead_pixel(self):
        points = np.array([[1, 1, 1], [0, 0, 0]], dtype=np.float32)
        orientations = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
        noshift = freud.pmft.PMFTXYZ(0.5, 0.5, 0.5, 3, 3, 3,
                                     shiftvec=[0, 0, 0])
        shift = freud.pmft.PMFTXYZ(0.5, 0.5, 0.5, 3, 3, 3,
                                   shiftvec=[1, 1, 1])

        for pm in [noshift, shift]:
            pm.compute(freud.box.Box.cube(3), points, orientations,
                       points, orientations, face_orientations=None)

        # Ignore warnings about NaNs
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Non-shifted pmft should have no non-inf valued voxels,
        # since the other point is outside the x/y/z max
        infcheck_noshift = np.isfinite(noshift.PMFT).sum()
        # Shifted pmft should have one non-inf valued voxel
        infcheck_shift = np.isfinite(shift.PMFT).sum()

        npt.assert_equal(infcheck_noshift, 0)
        npt.assert_equal(infcheck_shift, 1)


if __name__ == '__main__':
    unittest.main()
