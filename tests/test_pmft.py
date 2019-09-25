import numpy as np
import numpy.testing as npt
import freud
import unittest
import warnings
import util


class TestPMFTR12(unittest.TestCase):
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
        myPMFT = freud.pmft.PMFTR12(maxR, (nbinsR, nbinsT1, nbinsT2))
        myPMFT.accumulate(box, points, angles, points, angles)
        npt.assert_equal(myPMFT.box, freud.box.Box.square(boxSize))

        # Ensure expected errors are raised
        box = freud.box.Box.cube(boxSize)
        with self.assertRaises(ValueError):
            myPMFT.accumulate(box, points, angles, points, angles)

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

        listR = np.array([dr*(i+1/2) for i in range(nbinsR) if
                          dr*(i+1/2) < maxR])

        for i in range(nbinsT1):
            t = float(i) * dT1
            nextt = float(i + 1) * dT1
            listT1[i] = ((t + nextt) / 2.0)

        for i in range(nbinsT2):
            t = float(i) * dT2
            nextt = float(i + 1) * dT2
            listT2[i] = ((t + nextt) / 2.0)

        myPMFT = freud.pmft.PMFTR12(maxR, (nbinsR, nbinsT1, nbinsT2))

        # Compare expected bins to the info from pmft
        npt.assert_allclose(myPMFT.bin_centers[0], listR, atol=1e-3)
        npt.assert_allclose(myPMFT.bin_centers[1], listT1, atol=1e-3)
        npt.assert_allclose(myPMFT.bin_centers[2], listT2, atol=1e-3)

        npt.assert_equal(nbinsR, myPMFT.nbins[0])
        npt.assert_equal(nbinsT1, myPMFT.nbins[1])
        npt.assert_equal(nbinsT2, myPMFT.nbins[2])

    def test_attribute_access(self):
        boxSize = 16.0
        box = freud.box.Box.square(boxSize)
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.1, 0.0]],
                          dtype=np.float32)
        points.flags['WRITEABLE'] = False
        angles = np.array([0.0, np.pi/2], dtype=np.float32)
        angles.flags['WRITEABLE'] = False
        maxR = 5.23
        nbins = 10

        myPMFT = freud.pmft.PMFTR12(maxR, nbins)

        with self.assertRaises(AttributeError):
            myPMFT.PCF
        with self.assertRaises(AttributeError):
            myPMFT.bin_counts
        with self.assertRaises(AttributeError):
            myPMFT.box
        with self.assertRaises(AttributeError):
            myPMFT.PMFT

        myPMFT.accumulate(box, points, angles, points, angles)

        myPMFT.PCF
        myPMFT.bin_counts
        myPMFT.PMFT
        myPMFT.box
        npt.assert_equal(myPMFT.bin_counts.shape, (nbins, nbins, nbins))
        npt.assert_equal(myPMFT.PCF.shape, (nbins, nbins, nbins))
        npt.assert_equal(myPMFT.PMFT.shape, (nbins, nbins, nbins))

        myPMFT.reset()

        with self.assertRaises(AttributeError):
            myPMFT.PCF
        with self.assertRaises(AttributeError):
            myPMFT.bin_counts
        with self.assertRaises(AttributeError):
            myPMFT.box
        with self.assertRaises(AttributeError):
            myPMFT.PMFT

        myPMFT.compute(box, points, angles, points, angles)
        myPMFT.PCF
        myPMFT.bin_counts
        myPMFT.PMFT
        myPMFT.box

    def test_two_particles(self):
        boxSize = 16.0
        box = freud.box.Box.square(boxSize)
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.1, 0.0]],
                          dtype=np.float32)
        points.flags['WRITEABLE'] = False
        angles = np.array([0.0, np.pi/2], dtype=np.float32)
        angles.flags['WRITEABLE'] = False
        maxR = 5.23
        nbinsR = 10
        nbinsT1 = 20
        nbinsT2 = 30
        dr = (maxR / float(nbinsR))
        dT1 = (2.0 * np.pi / float(nbinsT1))
        dT2 = (2.0 * np.pi / float(nbinsT2))

        # calculation for array idxs
        def get_bin(point_i, point_j, angle_i, angle_j):
            delta_x = point_j - point_i
            r_bin = np.floor(np.linalg.norm(delta_x)/dr)
            delta_t1 = np.arctan2(delta_x[1], delta_x[0])
            delta_t2 = np.arctan2(-delta_x[1], -delta_x[0])
            t1_bin = np.floor(((angle_i - delta_t1) % (2. * np.pi))/dT1)
            t2_bin = np.floor(((angle_j - delta_t2) % (2. * np.pi))/dT2)
            return np.array([r_bin, t1_bin, t2_bin], dtype=np.int32)

        correct_bin_counts = np.zeros(shape=(nbinsR, nbinsT1, nbinsT2),
                                      dtype=np.int32)
        bins = get_bin(points[0], points[1], angles[0], angles[1])
        correct_bin_counts[bins[0], bins[1], bins[2]] = 1
        bins = get_bin(points[1], points[0], angles[1], angles[0])
        correct_bin_counts[bins[0], bins[1], bins[2]] = 1
        absoluteTolerance = 0.1

        r_max = maxR
        test_set = util.make_raw_query_nlist_test_set(
            box, points, points, 'ball', r_max, 0, True)
        for ts in test_set:
            myPMFT = freud.pmft.PMFTR12(maxR, (nbinsR, nbinsT1, nbinsT2))
            myPMFT.accumulate(box, ts[0], angles, nlist=ts[1])
            npt.assert_allclose(myPMFT.bin_counts, correct_bin_counts,
                                atol=absoluteTolerance)
            myPMFT.reset()
            myPMFT.compute(box, ts[0], angles, nlist=ts[1])
            npt.assert_allclose(myPMFT.bin_counts, correct_bin_counts,
                                atol=absoluteTolerance)

            myPMFT.compute(box, ts[0], angles, nlist=ts[1])
            npt.assert_allclose(myPMFT.bin_counts, correct_bin_counts,
                                atol=absoluteTolerance)

    def test_repr(self):
        maxR = 5.23
        nbins = 10
        myPMFT = freud.pmft.PMFTR12(maxR, nbins)
        self.assertEqual(str(myPMFT), str(eval(repr(myPMFT))))

    def test_points_ne_query_points(self):
        r_max = 2.3
        nbins = 10

        lattice_size = 10
        box = freud.box.Box.square(lattice_size*5)

        points, query_points = util.make_alternating_lattice(
            lattice_size, 0.01, 2)
        orientations = np.array([0]*len(points))
        query_orientations = np.array([0]*len(query_points))

        test_set = util.make_raw_query_nlist_test_set(
            box, points, query_points, "ball", r_max, 0, False)
        for ts in test_set:
            pmft = freud.pmft.PMFTR12(r_max, nbins)
            pmft.compute(box, ts[0],
                         orientations, query_points,
                         query_orientations, nlist=ts[1])

            self.assertEqual(np.count_nonzero(np.isinf(pmft.PMFT) == 0), 12)
            self.assertEqual(len(np.unique(pmft.PMFT)), 3)


class TestPMFTXYT(unittest.TestCase):
    def test_box(self):
        boxSize = 16.0
        box = freud.box.Box.square(boxSize)
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                          dtype=np.float32)
        angles = np.array([0.0, 0.0], dtype=np.float32)
        maxX = 3.6
        maxY = 4.2
        nbinsX = 20
        nbinsY = 30
        nbinsT = 40
        myPMFT = freud.pmft.PMFTXYT(maxX, maxY, (nbinsX, nbinsY, nbinsT))
        myPMFT.accumulate(box, points, angles, points, angles)
        npt.assert_equal(myPMFT.box, freud.box.Box.square(boxSize))

        # Ensure expected errors are raised
        box = freud.box.Box.cube(boxSize)
        with self.assertRaises(ValueError):
            myPMFT.accumulate(box, points, angles, points, angles)

    def test_bins(self):
        maxX = 3.6
        maxY = 4.2
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

        myPMFT = freud.pmft.PMFTXYT(maxX, maxY, (nbinsX, nbinsY, nbinsT))

        # Compare expected bins to the info from pmft
        npt.assert_allclose(myPMFT.bin_centers[0], listX, atol=1e-3)
        npt.assert_allclose(myPMFT.bin_centers[1], listY, atol=1e-3)
        npt.assert_allclose(myPMFT.bin_centers[2], listT, atol=1e-3)

        npt.assert_equal(nbinsX, myPMFT.nbins[0])
        npt.assert_equal(nbinsY, myPMFT.nbins[1])
        npt.assert_equal(nbinsT, myPMFT.nbins[2])

    def test_attribute_access(self):
        boxSize = 16.0
        box = freud.box.Box.square(boxSize)
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.1, 0.0]],
                          dtype=np.float32)
        angles = np.array([0.0, np.pi/2], dtype=np.float32)
        maxX = 3.6
        maxY = 4.2
        nbins = 20

        myPMFT = freud.pmft.PMFTXYT(maxX, maxY, nbins)

        with self.assertRaises(AttributeError):
            myPMFT.PCF
        with self.assertRaises(AttributeError):
            myPMFT.bin_counts
        with self.assertRaises(AttributeError):
            myPMFT.box
        with self.assertRaises(AttributeError):
            myPMFT.PMFT

        myPMFT.accumulate(box, points, angles, points, angles)

        myPMFT.PCF
        myPMFT.bin_counts
        myPMFT.PMFT
        myPMFT.box
        npt.assert_equal(myPMFT.bin_counts.shape, (nbins, nbins, nbins))
        npt.assert_equal(myPMFT.PCF.shape, (nbins, nbins, nbins))
        npt.assert_equal(myPMFT.PMFT.shape, (nbins, nbins, nbins))

        myPMFT.reset()

        with self.assertRaises(AttributeError):
            myPMFT.PCF
        with self.assertRaises(AttributeError):
            myPMFT.bin_counts
        with self.assertRaises(AttributeError):
            myPMFT.box
        with self.assertRaises(AttributeError):
            myPMFT.PMFT

        myPMFT.compute(box, points, angles, points, angles)
        myPMFT.PCF
        myPMFT.bin_counts
        myPMFT.PMFT
        myPMFT.box

    def test_two_particles(self):
        boxSize = 16.0
        box = freud.box.Box.square(boxSize)
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.1, 0.0]],
                          dtype=np.float32)
        angles = np.array([0.0, np.pi/2], dtype=np.float32)
        maxX = 3.6
        maxY = 4.2
        nbinsX = 20
        nbinsY = 30
        nbinsT = 40
        dx = (2.0 * maxX / float(nbinsX))
        dy = (2.0 * maxY / float(nbinsY))
        dT = (2.0 * np.pi / float(nbinsT))

        # calculation for array idxs
        def get_bin(point_i, point_j, angle_i, angle_j):
            delta_x = point_j - point_i
            rot_mat = np.array([[np.cos(-angle_i), -np.sin(-angle_i)],
                                [np.sin(-angle_i), np.cos(-angle_i)]])
            rot_delta_x = np.matmul(rot_mat, delta_x[:2])
            xy_bins = np.floor((rot_delta_x + [maxX, maxY]) /
                               [dx, dy]).astype(np.int32)
            angle_bin = np.floor(
                ((angle_j - np.arctan2(-delta_x[1], -delta_x[0])) %
                 (2. * np.pi)) / dT).astype(np.int32)
            return [xy_bins[0], xy_bins[1], angle_bin]

        correct_bin_counts = np.zeros(shape=(nbinsX, nbinsY, nbinsT),
                                      dtype=np.int32)
        bins = get_bin(points[0], points[1], angles[0], angles[1])
        correct_bin_counts[bins[0], bins[1], bins[2]] = 1
        bins = get_bin(points[1], points[0], angles[1], angles[0])
        correct_bin_counts[bins[0], bins[1], bins[2]] = 1
        absoluteTolerance = 0.1

        r_max = np.sqrt(maxX**2 + maxY**2)
        test_set = util.make_raw_query_nlist_test_set(
            box, points, points, 'ball', r_max, 0, True)
        for ts in test_set:
            myPMFT = freud.pmft.PMFTXYT(maxX, maxY, (nbinsX, nbinsY, nbinsT))
            myPMFT.accumulate(box, ts[0], angles, nlist=ts[1])
            npt.assert_allclose(myPMFT.bin_counts, correct_bin_counts,
                                atol=absoluteTolerance)
            myPMFT.reset()
            myPMFT.compute(box, ts[0], angles, nlist=ts[1])
            npt.assert_allclose(myPMFT.bin_counts, correct_bin_counts,
                                atol=absoluteTolerance)

            myPMFT.compute(box, ts[0], angles, nlist=ts[1])
            npt.assert_allclose(myPMFT.bin_counts, correct_bin_counts,
                                atol=absoluteTolerance)

    def test_repr(self):
        maxX = 3.0
        maxY = 4.0
        nbins = 20
        myPMFT = freud.pmft.PMFTXYT(maxX, maxY, nbins)
        self.assertEqual(str(myPMFT), str(eval(repr(myPMFT))))

    def test_points_ne_query_points(self):
        x_max = 2.5
        y_max = 2.5
        n_x = 10
        n_y = 10
        n_t = 4

        lattice_size = 10
        box = freud.box.Box.square(lattice_size*5)

        points, query_points = util.make_alternating_lattice(
            lattice_size, 0.01, 2)
        orientations = np.array([0]*len(points))
        query_orientations = np.array([0]*len(query_points))

        r_max = np.sqrt(x_max**2 + y_max**2)
        test_set = util.make_raw_query_nlist_test_set(
            box, points, query_points, 'ball', r_max, 0, False)

        for ts in test_set:
            pmft = freud.pmft.PMFTXYT(x_max, y_max, (n_x, n_y, n_t))
            pmft.compute(box, ts[0],
                         orientations, query_points,
                         query_orientations, nlist=ts[1])

            # when rotated slightly, for each ref point, each quadrant
            # (corresponding to two consecutive bins) should contain 3 points.
            for i in range(n_t):
                self.assertEqual(
                    np.count_nonzero(np.isinf(pmft.PMFT[..., i]) == 0), 3)

            self.assertEqual(len(np.unique(pmft.PMFT)), 2)


class TestPMFTXY2D(unittest.TestCase):
    def test_box(self):
        boxSize = 16.0
        box = freud.box.Box.square(boxSize)
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                          dtype=np.float32)
        angles = np.array([0.0, 0.0], dtype=np.float32)
        maxX = 3.6
        maxY = 4.2
        nbinsX = 100
        nbinsY = 110
        myPMFT = freud.pmft.PMFTXY2D(maxX, maxY, (nbinsX, nbinsY))
        myPMFT.accumulate(box, points, angles, points)
        npt.assert_equal(myPMFT.box, freud.box.Box.square(boxSize))

        # Ensure expected errors are raised
        box = freud.box.Box.cube(boxSize)
        with self.assertRaises(ValueError):
            myPMFT.accumulate(box, points, angles, points)

    def test_bins(self):
        maxX = 3.6
        maxY = 4.2
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

        myPMFT = freud.pmft.PMFTXY2D(maxX, maxY, (nbinsX, nbinsY))

        # Compare expected bins to the info from pmft
        npt.assert_allclose(myPMFT.bin_centers[0], listX, atol=1e-3)
        npt.assert_allclose(myPMFT.bin_centers[1], listY, atol=1e-3)

        npt.assert_equal(nbinsX, myPMFT.nbins[0])
        npt.assert_equal(nbinsY, myPMFT.nbins[1])

    def test_attribute_access(self):
        boxSize = 16.0
        box = freud.box.Box.square(boxSize)
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                          dtype=np.float32)
        angles = np.array([0.0, 0.0], dtype=np.float32)
        maxX = 3.6
        maxY = 4.2
        nbins = 100

        myPMFT = freud.pmft.PMFTXY2D(maxX, maxY, nbins)

        with self.assertRaises(AttributeError):
            myPMFT.PCF
        with self.assertRaises(AttributeError):
            myPMFT.bin_counts
        with self.assertRaises(AttributeError):
            myPMFT.box
        with self.assertRaises(AttributeError):
            myPMFT.PMFT

        myPMFT.accumulate(box, points, angles, points)

        myPMFT.PCF
        myPMFT.bin_counts
        myPMFT.PMFT
        myPMFT.box
        npt.assert_equal(myPMFT.bin_counts.shape, (nbins, nbins))
        npt.assert_equal(myPMFT.PCF.shape, (nbins, nbins))
        npt.assert_equal(myPMFT.PMFT.shape, (nbins, nbins))

        myPMFT.reset()

        with self.assertRaises(AttributeError):
            myPMFT.PCF
        with self.assertRaises(AttributeError):
            myPMFT.bin_counts
        with self.assertRaises(AttributeError):
            myPMFT.box
        with self.assertRaises(AttributeError):
            myPMFT.PMFT

        myPMFT.compute(box, points, angles, points)
        myPMFT.PCF
        myPMFT.bin_counts
        myPMFT.PMFT
        myPMFT.box

    def test_two_particles(self):
        boxSize = 16.0
        box = freud.box.Box.square(boxSize)
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                          dtype=np.float32)
        angles = np.array([0.0, 0.0], dtype=np.float32)
        maxX = 3.6
        maxY = 4.2
        nbinsX = 100
        nbinsY = 110
        dx = (2.0 * maxX / float(nbinsX))
        dy = (2.0 * maxY / float(nbinsY))

        correct_bin_counts = np.zeros(shape=(nbinsX, nbinsY), dtype=np.int32)

        # calculation for array idxs
        def get_bin(point_i, point_j):
            return np.floor((point_i - point_j + [maxX, maxY, 0]) /
                            [dx, dy, 1]).astype(np.int32)[:2]

        bins = get_bin(points[0], points[1])
        correct_bin_counts[bins[0], bins[1]] = 1
        bins = get_bin(points[1], points[0])
        correct_bin_counts[bins[0], bins[1]] = 1
        absoluteTolerance = 0.1

        r_max = np.sqrt(maxX**2 + maxY**2)
        test_set = util.make_raw_query_nlist_test_set(
            box, points, points, 'ball', r_max, 0, True)
        for ts in test_set:
            myPMFT = freud.pmft.PMFTXY2D(maxX, maxY, (nbinsX, nbinsY))
            myPMFT.accumulate(box, ts[0], angles, nlist=ts[1])
            npt.assert_allclose(myPMFT.bin_counts, correct_bin_counts,
                                atol=absoluteTolerance)
            myPMFT.reset()
            myPMFT.compute(box, ts[0], angles, nlist=ts[1])
            npt.assert_allclose(myPMFT.bin_counts, correct_bin_counts,
                                atol=absoluteTolerance)

            myPMFT.compute(box, ts[0], angles, nlist=ts[1])
            npt.assert_allclose(myPMFT.bin_counts, correct_bin_counts,
                                atol=absoluteTolerance)

    def test_repr(self):
        maxX = 3.0
        maxY = 4.0
        nbins = 100
        myPMFT = freud.pmft.PMFTXY2D(maxX, maxY, nbins)
        self.assertEqual(str(myPMFT), str(eval(repr(myPMFT))))

    def test_repr_png(self):
        boxSize = 16.0
        box = freud.box.Box.square(boxSize)
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                          dtype=np.float32)
        angles = np.array([0.0, 0.0], dtype=np.float32)
        maxX = 3.6
        maxY = 4.2
        nbinsX = 100
        nbinsY = 110
        myPMFT = freud.pmft.PMFTXY2D(maxX, maxY, (nbinsX, nbinsY))

        with self.assertRaises(AttributeError):
            myPMFT.plot()
        self.assertEqual(myPMFT._repr_png_(), None)

        myPMFT.accumulate(box, points, angles, points)
        myPMFT._repr_png_()

    def test_points_ne_query_points(self):
        x_max = 2.5
        y_max = 2.5
        nbins = 20

        lattice_size = 10
        box = freud.box.Box.square(lattice_size*5)

        points, query_points = util.make_alternating_lattice(
            lattice_size, 0.01, 2)

        orientations = np.array([0]*len(points))

        r_max = np.sqrt(x_max**2 + y_max**2)
        test_set = util.make_raw_query_nlist_test_set(
            box, points, query_points, 'ball', r_max, 0, False)

        for ts in test_set:
            pmft = freud.pmft.PMFTXY2D(x_max, y_max, nbins)
            pmft.compute(
                box, ts[0], orientations, query_points, ts[1])

            self.assertEqual(np.count_nonzero(np.isinf(pmft.PMFT) == 0), 12)
            self.assertEqual(len(np.unique(pmft.PMFT)), 2)

    def test_query_args_nn(self):
        """Test that using nn based query args works."""
        boxSize = 8
        box = freud.box.Box.square(boxSize)
        points = np.array([[0, 0, 0]],
                          dtype=np.float32)
        query_points = np.array([[1.1, 0.0, 0.0],
                                [-1.2, 0.0, 0.0],
                                [0.0, 1.3, 0.0],
                                [0.0, -1.4, 0.0]],
                                dtype=np.float32)
        angles = np.array([0.0]*points.shape[0], dtype=np.float32)
        query_angles = np.array([0.0]*query_points.shape[0], dtype=np.float32)

        max_width = 3
        nbins = 3
        pmft = freud.pmft.PMFTXY2D(max_width, max_width, nbins)
        pmft.compute(box, points, angles, query_points,
                     query_args={'mode': 'nearest', 'num_neighbors': 1})
        # Now every point in query_points will find the origin as a neighbor.
        npt.assert_array_equal(
            pmft.bin_counts,
            [[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]])
        # Now there will be only one neighbor for the single point.
        pmft.compute(box, query_points, query_angles, points,
                     query_args={'mode': 'nearest', 'num_neighbors': 1})
        npt.assert_array_equal(
            pmft.bin_counts,
            [[0, 1, 0],
             [0, 0, 0],
             [0, 0, 0]])


class TestPMFTXYZ(unittest.TestCase):
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
        myPMFT = freud.pmft.PMFTXYZ(maxX, maxY, maxZ, (nbinsX, nbinsY, nbinsZ))
        myPMFT.accumulate(box, points, orientations, points, orientations)
        npt.assert_equal(myPMFT.box, freud.box.Box.cube(boxSize))

        # Ensure expected errors are raised
        box = freud.box.Box.square(boxSize)
        with self.assertRaises(ValueError):
            myPMFT.accumulate(box, points, orientations, points, orientations)

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

        myPMFT = freud.pmft.PMFTXYZ(maxX, maxY, maxZ, (nbinsX, nbinsY, nbinsZ))

        # Compare expected bins to the info from pmft
        npt.assert_allclose(myPMFT.bin_centers[0], listX, atol=1e-3)
        npt.assert_allclose(myPMFT.bin_centers[1], listY, atol=1e-3)
        npt.assert_allclose(myPMFT.bin_centers[2], listZ, atol=1e-3)

        npt.assert_equal(nbinsX, myPMFT.nbins[0])
        npt.assert_equal(nbinsY, myPMFT.nbins[1])
        npt.assert_equal(nbinsZ, myPMFT.nbins[2])

    def test_attribute_access(self):
        boxSize = 25.0
        box = freud.box.Box.cube(boxSize)
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                          dtype=np.float32)
        orientations = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
        maxX = 5.23
        maxY = 6.23
        maxZ = 7.23
        nbinsX = nbinsY = nbinsZ = 100

        myPMFT = freud.pmft.PMFTXYZ(maxX, maxY, maxZ, nbinsX)

        with self.assertRaises(AttributeError):
            myPMFT.PCF
        with self.assertRaises(AttributeError):
            myPMFT.bin_counts
        with self.assertRaises(AttributeError):
            myPMFT.box
        with self.assertRaises(AttributeError):
            myPMFT.PMFT

        myPMFT.accumulate(box, points, orientations, points, orientations)

        myPMFT.PCF
        myPMFT.bin_counts
        myPMFT.PMFT
        myPMFT.box
        npt.assert_equal(myPMFT.bin_counts.shape, (nbinsX, nbinsY, nbinsZ))
        npt.assert_equal(myPMFT.PCF.shape, (nbinsX, nbinsY, nbinsZ))
        npt.assert_equal(myPMFT.PMFT.shape, (nbinsX, nbinsY, nbinsZ))

        myPMFT.reset()

        with self.assertRaises(AttributeError):
            myPMFT.PCF
        with self.assertRaises(AttributeError):
            myPMFT.bin_counts
        with self.assertRaises(AttributeError):
            myPMFT.box
        with self.assertRaises(AttributeError):
            myPMFT.PMFT

        myPMFT.compute(box, points, orientations, points, orientations)
        myPMFT.PCF
        myPMFT.bin_counts
        myPMFT.PMFT
        myPMFT.box

    def test_two_particles(self):
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
        dx = (2.0 * maxX / float(nbinsX))
        dy = (2.0 * maxY / float(nbinsY))
        dz = (2.0 * maxZ / float(nbinsZ))

        correct_bin_counts = np.zeros(shape=(nbinsX, nbinsY, nbinsZ),
                                      dtype=np.int32)

        # calculation for array idxs
        def get_bin(point_i, point_j):
            return np.floor((point_i - point_j + [maxX, maxY, maxZ]) /
                            [dx, dy, dz]).astype(np.int32)

        bins = get_bin(points[0], points[1])
        correct_bin_counts[bins[0], bins[1], bins[2]] = 1
        bins = get_bin(points[1], points[0])
        correct_bin_counts[bins[0], bins[1], bins[2]] = 1
        absoluteTolerance = 0.1

        r_max = np.sqrt(maxX**2 + maxY**2 + maxZ**2)
        test_set = util.make_raw_query_nlist_test_set(
            box, points, points, 'ball', r_max, 0, True)
        for ts in test_set:
            myPMFT = freud.pmft.PMFTXYZ(
                maxX, maxY, maxZ, (nbinsX, nbinsY, nbinsZ))
            myPMFT.accumulate(box, ts[0], orientations, nlist=ts[1])
            npt.assert_allclose(myPMFT.bin_counts, correct_bin_counts,
                                atol=absoluteTolerance)
            myPMFT.reset()
            myPMFT.compute(box, ts[0], orientations, nlist=ts[1])
            npt.assert_allclose(myPMFT.bin_counts, correct_bin_counts,
                                atol=absoluteTolerance)

            # Test face orientations, shape (N_faces, 4)
            face_orientations = np.array([[1., 0., 0., 0.]])
            myPMFT.compute(box, ts[0], orientations, nlist=ts[1],
                           face_orientations=face_orientations)
            npt.assert_allclose(myPMFT.bin_counts, correct_bin_counts,
                                atol=absoluteTolerance)
            # Test face orientations, shape (1, N_faces, 4)
            face_orientations = np.array([[[1., 0., 0., 0.]]])
            myPMFT.compute(box, ts[0], orientations, nlist=ts[1],
                           face_orientations=face_orientations)
            npt.assert_allclose(myPMFT.bin_counts, correct_bin_counts,
                                atol=absoluteTolerance)
            # Test face orientations, shape (N_particles, N_faces, 4)
            face_orientations = np.array([[[1., 0., 0., 0.]],
                                          [[1., 0., 0., 0.]]])
            myPMFT.compute(box, ts[0], orientations, nlist=ts[1],
                           face_orientations=face_orientations)
            npt.assert_allclose(myPMFT.bin_counts, correct_bin_counts,
                                atol=absoluteTolerance)

            myPMFT.compute(box, ts[0], orientations, nlist=ts[1])
            npt.assert_allclose(myPMFT.bin_counts, correct_bin_counts,
                                atol=absoluteTolerance)

    def test_shift_two_particles_dead_pixel(self):
        points = np.array([[1, 1, 1], [0, 0, 0]], dtype=np.float32)
        orientations = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
        noshift = freud.pmft.PMFTXYZ(0.5, 0.5, 0.5, 3, shiftvec=[0, 0, 0])
        shift = freud.pmft.PMFTXYZ(0.5, 0.5, 0.5, 3, shiftvec=[1, 1, 1])

        for pm in [noshift, shift]:
            pm.compute(freud.box.Box.cube(3), points, orientations)

        # Ignore warnings about NaNs
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Non-shifted pmft should have no non-inf valued voxels,
        # since the other point is outside the x/y/z max
        infcheck_noshift = np.isfinite(noshift.PMFT).sum()
        # Shifted pmft should have one non-inf valued voxel
        infcheck_shift = np.isfinite(shift.PMFT).sum()

        npt.assert_equal(infcheck_noshift, 0)
        npt.assert_equal(infcheck_shift, 1)

    def test_repr(self):
        maxX = 5.23
        maxY = 6.23
        maxZ = 7.23
        nbinsX = 100
        nbinsY = 110
        nbinsZ = 120
        myPMFT = freud.pmft.PMFTXYZ(maxX, maxY, maxZ, (nbinsX, nbinsY, nbinsZ))
        self.assertEqual(str(myPMFT), str(eval(repr(myPMFT))))

    def test_query_args_nn(self):
        """Test that using nn based query args works."""
        boxSize = 8
        box = freud.box.Box.cube(boxSize)
        points = np.array([[0, 0, 0]],
                          dtype=np.float32)
        query_points = np.array([[1.1, 0.0, 0.0],
                                [-1.2, 0.0, 0.0],
                                [0.0, 1.3, 0.0],
                                [0.0, -1.4, 0.0],
                                [0.0, 0.0, 1.5],
                                [0.0, 0.0, -1.6]],
                                dtype=np.float32)
        angles = np.array([[1.0, 0.0, 0.0, 0.0]]*points.shape[0],
                          dtype=np.float32)
        query_angles = np.array([[1.0, 0.0, 0.0, 0.0]]*query_points.shape[0],
                                dtype=np.float32)

        max_width = 3
        nbins = 3
        pmft = freud.pmft.PMFTXYZ(max_width, max_width, max_width, nbins)
        pmft.compute(box, points, angles, query_points,
                     query_args={'mode': 'nearest', 'num_neighbors': 1})

        # Now every point in query_points will find the origin as a neighbor.
        npt.assert_array_equal(
            pmft.bin_counts,
            [[[0, 0, 0],
              [0, 1, 0],
              [0, 0, 0]],
             [[0, 1, 0],
              [1, 0, 1],
              [0, 1, 0]],
             [[0, 0, 0],
              [0, 1, 0],
              [0, 0, 0]]])

        pmft.compute(box, query_points, query_angles, points,
                     query_args={'mode': 'nearest', 'num_neighbors': 1})
        # The only nonzero bin is in the left bin for x, but the center for
        # everything else (0 distance in y and z).
        self.assertEqual(pmft.bin_counts[0, 1, 1], 1)
        self.assertEqual(np.sum(pmft.bin_counts), 1)
        self.assertTrue(np.all(pmft.bin_counts >= 0))


if __name__ == '__main__':
    unittest.main()
