import numpy.testing as npt
import numpy as np
import freud
import unittest
from util import make_box_and_random_points


class TestAngularSeparationGlobal(unittest.TestCase):
    def test_getN(self):
        boxlen = 10
        N = 500

        box, points = make_box_and_random_points(boxlen, N, True)
        _, query_points = make_box_and_random_points(boxlen, N//3, True)

        ang = freud.environment.AngularSeparationGlobal()

        # test access
        with self.assertRaises(AttributeError):
            ang.angles

    def test_compute(self):
        # Going to make sure that the use of equivalent_orientations captures
        # both of the global reference orientations
        global_ors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        equivalent_orientations = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, -1, 0, 0]],
            dtype=np.float32)

        ors = [[1, 0, 0, 0]]
        ors.append([0, 1, 0, 0])
        # The following two quaternions correspond to rotations of the above
        # by pi/16
        ors.append([0.99518473, 0., 0., 0.09801714])
        ors.append([0., 0.99518473, -0.09801714, 0.])

        ors = np.asarray(ors, dtype=np.float32)

        ang = freud.environment.AngularSeparationGlobal()
        ang.compute(global_ors, ors, equivalent_orientations)

        # Each orientation should be either equal to or pi/16 away from the
        # global reference quaternion
        for i in [0, 1]:
            for j in [0, 1]:
                npt.assert_allclose(ang.angles[i][j], 0, atol=1e-6)
        for i in [2, 3]:
            for j in [0, 1]:
                npt.assert_allclose(ang.angles[i][j], np.pi/16,
                                    atol=1e-6)

    def test_repr(self):
        ang = freud.environment.AngularSeparationGlobal()
        self.assertEqual(str(ang), str(eval(repr(ang))))


class TestAngularSeparationNeighbor(unittest.TestCase):
    def test_getN(self):
        boxlen = 10
        N = 500

        box, points = make_box_and_random_points(boxlen, N, True)
        _, query_points = make_box_and_random_points(boxlen, N//3, True)

        ang = freud.environment.AngularSeparationNeighbor()

        # test access
        with self.assertRaises(AttributeError):
            ang.angles

    def test_nlist(self):
        """Check that the internally generated NeighborList is correct."""
        boxlen = 4
        num_neighbors = 1
        r_guess = 2

        box = freud.box.Box.square(boxlen)

        # Create three points in a line.
        points = np.asarray([[0, 0, 0], [1, 0, 0], [1.5, 0, 0]],
                            dtype=np.float32)
        # Use two separate orientations. The second orientation is a pi/3
        # rotation from the identity quaternion
        ors = np.asarray([[1, 0, 0, 0],
                          [np.cos(np.pi/6), np.sin(np.pi/6), 0, 0],
                          [np.cos(np.pi/6), np.sin(np.pi/6), 0, 0]],
                         dtype=np.float32)

        equivalent_orientations = np.asarray([[1, 0, 0, 0], [-1, 0, 0, 0]],
                                             dtype=np.float32)

        ang = freud.environment.AngularSeparationNeighbor()
        qargs = dict(num_neighbors=num_neighbors, r_guess=r_guess,
                     exclude_ii=True)
        ang.compute((box, points), ors,
                    equiv_orientations=equivalent_orientations,
                    neighbors=qargs)

        aq = freud.locality.AABBQuery(box, points)
        nlist = aq.query(points, qargs).toNeighborList()

        npt.assert_array_equal(nlist[:], ang.nlist[:])

    def test_compute(self):
        boxlen = 4
        num_neighbors = 1
        r_guess = 2

        box = freud.box.Box.square(boxlen)

        # Create three points in a line.
        points = np.asarray([[0, 0, 0], [1, 0, 0], [1.5, 0, 0]],
                            dtype=np.float32)
        # Use two separate orientations. The second orientation is a pi/3
        # rotation from the identity quaternion
        ors = np.asarray([[1, 0, 0, 0],
                          [np.cos(np.pi/6), np.sin(np.pi/6), 0, 0],
                          [np.cos(np.pi/6), np.sin(np.pi/6), 0, 0]],
                         dtype=np.float32)

        equivalent_orientations = np.asarray([[1, 0, 0, 0], [-1, 0, 0, 0]],
                                             dtype=np.float32)

        ang = freud.environment.AngularSeparationNeighbor()
        ang.compute((box, points), ors,
                    equiv_orientations=equivalent_orientations,
                    neighbors=dict(num_neighbors=num_neighbors,
                                   r_guess=r_guess))

        # Should find that the angular separation between the first particle
        # and its neighbor is pi/3. The second particle's nearest neighbor will
        # have the same orientation.
        npt.assert_allclose(ang.angles[0], np.pi/3, atol=1e-6)
        npt.assert_allclose(ang.angles[1], 0, atol=1e-6)

    def test_repr(self):
        ang = freud.environment.AngularSeparationNeighbor()
        self.assertEqual(str(ang), str(eval(repr(ang))))


if __name__ == '__main__':
    unittest.main()
