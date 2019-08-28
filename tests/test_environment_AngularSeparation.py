import numpy.testing as npt
import numpy as np
import freud
import rowan
import unittest
from util import make_box_and_random_points


class TestAngularSeparation(unittest.TestCase):
    def test_getN(self):
        boxlen = 10
        N = 500
        num_neighbors = 8
        r_max = 3

        box, points = make_box_and_random_points(boxlen, N, True)
        _, query_points = make_box_and_random_points(boxlen, N//3, True)
        ors = rowan.random.rand(N)
        query_ors = rowan.random.rand(N//3)

        ang = freud.environment.AngularSeparation(r_max, num_neighbors)

        # test access
        with self.assertRaises(AttributeError):
            ang.neighbor_angles
        with self.assertRaises(AttributeError):
            ang.global_angles
        with self.assertRaises(AttributeError):
            ang.n_points
        with self.assertRaises(AttributeError):
            ang.n_global
        with self.assertRaises(AttributeError):
            ang.n_query_points

        # defaulted_nlist = freud.locality.make_default_nlist_nn(
        #     box, points, query_points, num_neighbors,
        #     None, False, r_max)
        # for i, j in defaulted_nlist[0]:
        #     print(i, j)
        ang.computeNeighbor(box, points, ors, query_points, query_ors)
        self.assertEqual(ang.n_points, N)
        self.assertEqual(ang.n_query_points, N//3)

    def test_getNGlobal(self):
        N = 500
        num_neighbors = 8
        r_max = 3

        ors = rowan.random.rand(N)
        equiv_quats = np.asarray([[1, 0, 0, 0]], dtype=np.float32)
        global_ors = np.array([[1, 0, 0, 0]], dtype=np.float32)

        ang = freud.environment.AngularSeparation(r_max, num_neighbors)
        ang.computeGlobal(global_ors, ors, equiv_quats)
        npt.assert_equal(ang.n_global, 1)

    def test_get_num_points(self):
        boxlen = 10
        N = 500
        num_neighbors = 8
        r_max = 3

        box, points = make_box_and_random_points(boxlen, N, True)
        ors = rowan.random.rand(N)

        ang = freud.environment.AngularSeparation(r_max, num_neighbors)
        ang.computeNeighbor(box, points, ors, query_orientations=ors)
        npt.assert_equal(ang.n_points, N)

    def test_compute_neighbors(self):
        boxlen = 4
        num_neighbors = 1
        r_max = 2

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

        equiv_quats = np.asarray([[1, 0, 0, 0], [-1, 0, 0, 0]],
                                 dtype=np.float32)

        ang = freud.environment.AngularSeparation(r_max, num_neighbors)
        ang.computeNeighbor(box, points, ors, equiv_orientations=equiv_quats)

        # test access
        ang.neighbor_angles
        ang.n_points
        ang.n_query_points
        with self.assertRaises(AttributeError):
            ang.global_angles
        with self.assertRaises(AttributeError):
            ang.n_global

        # Should find that the angular separation between the first particle
        # and its neighbor is pi/3. The second particle's nearest neighbor will
        # have the same orientation.
        npt.assert_allclose(ang.neighbor_angles[0], np.pi/3, atol=1e-6)
        npt.assert_allclose(ang.neighbor_angles[1], 0, atol=1e-6)

    def test_compute_global(self):
        num_neighbors = 1
        r_max = 2

        # Going to make sure that the use of equiv_quats captures both of the
        # global reference orientations
        global_ors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        equiv_quats = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                [-1, 0, 0, 0], [0, -1, 0, 0]],
                               dtype=np.float32)

        ors = [[1, 0, 0, 0]]
        ors.append([0, 1, 0, 0])
        # The following two quaternions correspond to rotations of the above
        # by pi/16
        ors.append([0.99518473, 0., 0., 0.09801714])
        ors.append([0., 0.99518473, -0.09801714, 0.])

        ors = np.asarray(ors, dtype=np.float32)

        ang = freud.environment.AngularSeparation(r_max, num_neighbors)
        ang.computeGlobal(global_ors, ors, equiv_quats)

        # test access
        ang.global_angles
        ang.n_points
        ang.n_global
        with self.assertRaises(AttributeError):
            ang.neighbor_angles
        with self.assertRaises(AttributeError):
            ang.n_query_points

        # Each orientation should be either equal to or pi/16 away from the
        # global reference quaternion
        for i in [0, 1]:
            for j in [0, 1]:
                npt.assert_allclose(ang.global_angles[i][j], 0, atol=1e-6)
        for i in [2, 3]:
            for j in [0, 1]:
                npt.assert_allclose(ang.global_angles[i][j], np.pi/16,
                                    atol=1e-6)

    def test_repr(self):
        ang = freud.environment.AngularSeparation(3, 12)
        self.assertEqual(str(ang), str(eval(repr(ang))))


if __name__ == '__main__':
    unittest.main()
