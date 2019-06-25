import numpy.testing as npt
import numpy as np
import freud
import unittest


def quatRandom():
    """Returns a random quaternion culled from a uniform distribution on the
    surface of a 3-sphere. Uses the Marsaglia (1972) method (a la HOOMD).
    Note that generating a random rotation via a random angle about a random
    axis of rotation is INCORRECT. See K. Shoemake, "Uniform Random Rotations,"
    1992, for a nice explanation for this.

    The output quaternion is an array of four numbers: [q0, q1, q2, q3]"""

    # np.random.uniform(low, high) gives a number from the interval [low, high)
    v1 = np.random.uniform(-1, 1)
    v2 = np.random.uniform(-1, 1)
    v3 = np.random.uniform(-1, 1)
    v4 = np.random.uniform(-1, 1)

    s1 = v1*v1 + v2*v2
    s2 = v3*v3 + v4*v4

    while (s1 >= 1.):
        v1 = np.random.uniform(-1, 1)
        v2 = np.random.uniform(-1, 1)
        s1 = v1*v1 + v2*v2

    while (s2 >= 1. or s2 == 0.):
        v3 = np.random.uniform(-1, 1)
        v4 = np.random.uniform(-1, 1)
        s2 = v3*v3 + v4*v4

    s3 = np.sqrt((1.-s1)/s2)

    return np.array([v1, v2, v3*s3, v4*s3])


class TestAngularSeparation(unittest.TestCase):
    def test_getNP(self):
        boxlen = 10
        N = 500
        num_neigh = 8
        rmax = 3

        box = freud.box.Box.square(boxlen)

        np.random.seed(0)
        points = np.asarray(np.random.uniform(-boxlen/2, boxlen/2, (N, 3)),
                            dtype=np.float32)
        ors = []
        for i in range(N):
            ors.append(quatRandom())

        ors = np.asarray(ors, dtype=np.float32)
        equiv_quats = np.asarray([[1, 0, 0, 0]], dtype=np.float32)

        ang = freud.environment.AngularSeparation(rmax, num_neigh)

        # test access
        with self.assertRaises(AttributeError):
            ang.neighbor_angles
        with self.assertRaises(AttributeError):
            ang.global_angles
        with self.assertRaises(AttributeError):
            ang.n_p
        with self.assertRaises(AttributeError):
            ang.n_global
        with self.assertRaises(AttributeError):
            ang.n_ref

        ang.computeNeighbor(box, ors, ors, points, points, equiv_quats)
        npt.assert_equal(ang.n_p, N)

    def test_getNGlobal(self):
        N = 500
        num_neigh = 8
        rmax = 3

        ors = []
        for i in range(N):
            ors.append(quatRandom())

        ors = np.asarray(ors, dtype=np.float32)
        equiv_quats = np.asarray([[1, 0, 0, 0]], dtype=np.float32)
        global_ors = np.array([[1, 0, 0, 0]], dtype=np.float32)

        ang = freud.environment.AngularSeparation(rmax, num_neigh)
        ang.computeGlobal(global_ors, ors, equiv_quats)
        npt.assert_equal(ang.n_global, 1)

    def test_getNReference(self):
        boxlen = 10
        N = 500
        num_neigh = 8
        rmax = 3

        box = freud.box.Box.square(boxlen)

        np.random.seed(0)
        points = np.asarray(np.random.uniform(-boxlen/2, boxlen/2, (N, 3)),
                            dtype=np.float32)
        ors = []
        for i in range(N):
            ors.append(quatRandom())

        ors = np.asarray(ors, dtype=np.float32)
        equiv_quats = np.asarray([[1, 0, 0, 0]], dtype=np.float32)

        ang = freud.environment.AngularSeparation(rmax, num_neigh)
        ang.computeNeighbor(box, ors, ors, points, points, equiv_quats)
        print(ang.neighbor_angles)
        npt.assert_equal(ang.n_ref, N)

    def test_compute_neighbors(self):
        boxlen = 4
        num_neigh = 1
        rmax = 2

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

        ang = freud.environment.AngularSeparation(rmax, num_neigh)
        ang.computeNeighbor(box, ors, ors, points, points, equiv_quats)

        # test access
        ang.neighbor_angles
        ang.n_p
        ang.n_ref
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
        num_neigh = 1
        rmax = 2

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

        ang = freud.environment.AngularSeparation(rmax, num_neigh)
        ang.computeGlobal(global_ors, ors, equiv_quats)

        # test access
        ang.global_angles
        ang.n_p
        ang.n_global
        with self.assertRaises(AttributeError):
            ang.neighbor_angles
        with self.assertRaises(AttributeError):
            ang.n_ref

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
