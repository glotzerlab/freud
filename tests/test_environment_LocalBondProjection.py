import unittest
import numpy.testing as npt
import numpy as np
import random
import freud
from util import makeBoxAndRandomPoints


random.seed(0)


# Returns a random quaternion culled from a uniform distribution on the surface
# of a 3-sphere. Uses the MARSAGLIA (1972) method (a la hoomd) NOTE THAT
# generating a random rotation via a random angle about a random axis of
# rotation is INCORRECT. See K. Shoemake, "Uniform Random Rotations," 1992,
# for a nice explanation for this. Output quat is an array of four numbers:
# [q0, q1, q2, q3]
def quatRandom():
    # random.uniform(a,b) gives number in [a,b]
    v1 = random.uniform(-1, 1)
    v2 = random.uniform(-1, 1)
    v3 = random.uniform(-1, 1)
    v4 = random.uniform(-1, 1)

    s1 = v1*v1 + v2*v2
    s2 = v3*v3 + v4*v4

    while (s1 >= 1.):
        v1 = random.uniform(-1, 1)
        v2 = random.uniform(-1, 1)
        s1 = v1*v1 + v2*v2

    while (s2 >= 1. or s2 == 0.):
        v3 = random.uniform(-1, 1)
        v4 = random.uniform(-1, 1)
        s2 = v3*v3 + v4*v4

    s3 = np.sqrt((1.-s1)/s2)

    return np.array([v1, v2, v3*s3, v4*s3])


class TestLocalBondProjection(unittest.TestCase):
    def test_num_points(self):
        boxlen = 10
        N = 500
        num_neigh = 8
        rmax = 3

        N_query = N//3

        box, points = makeBoxAndRandomPoints(boxlen, N, True)
        _, query_points = makeBoxAndRandomPoints(boxlen, N_query, True)

        ors = []
        for i in range(N):
            ors.append(quatRandom())

        ors = np.asarray(ors, dtype=np.float32)
        proj_vecs = np.asarray([[0, 0, 1]])

        ang = freud.environment.LocalBondProjection(rmax, num_neigh)
        ang.compute(box, proj_vecs, points, ors, query_points)
        self.assertEqual(ang.num_points, N)
        self.assertEqual(ang.num_query_points, N_query)

    def test_num_proj_vectors(self):
        boxlen = 10
        N = 500
        num_neigh = 8
        rmax = 3

        box, points = makeBoxAndRandomPoints(boxlen, N, True)

        ors = []
        for i in range(N):
            ors.append(quatRandom())

        ors = np.asarray(ors, dtype=np.float32)
        proj_vecs = np.asarray([[0, 0, 1]])

        ang = freud.environment.LocalBondProjection(rmax, num_neigh)
        ang.compute(box, proj_vecs, points, ors)
        npt.assert_equal(ang.num_proj_vectors, 1)

    def test_box(self):
        boxlen = 10
        N = 500
        num_neigh = 8
        rmax = 3

        box, points = makeBoxAndRandomPoints(boxlen, N)

        ors = []
        for i in range(N):
            ors.append(quatRandom())

        ors = np.asarray(ors, dtype=np.float32)
        proj_vecs = np.asarray([[0, 0, 1]])

        ang = freud.environment.LocalBondProjection(rmax, num_neigh)
        ang.compute(box, proj_vecs, points, ors)

        npt.assert_equal(ang.box.Lx, boxlen)
        npt.assert_equal(ang.box.Ly, boxlen)
        npt.assert_equal(ang.box.Lz, boxlen)
        npt.assert_equal(ang.box.xy, 0)
        npt.assert_equal(ang.box.xz, 0)
        npt.assert_equal(ang.box.yz, 0)

    def test_attribute_access(self):
        boxlen = 10
        N = 100
        num_neigh = 8
        rmax = 3

        box, points = makeBoxAndRandomPoints(boxlen, N, True)

        ors = []
        for i in range(N):
            ors.append(quatRandom())

        ors = np.asarray(ors, dtype=np.float32)
        proj_vecs = np.asarray([[0, 0, 1]])

        ang = freud.environment.LocalBondProjection(rmax, num_neigh)

        with self.assertRaises(AttributeError):
            ang.nlist
        with self.assertRaises(AttributeError):
            ang.projections
        with self.assertRaises(AttributeError):
            ang.normed_projections
        with self.assertRaises(AttributeError):
            ang.num_points
        with self.assertRaises(AttributeError):
            ang.num_points
        with self.assertRaises(AttributeError):
            ang.num_proj_vectors
        with self.assertRaises(AttributeError):
            ang.box

        ang.compute(box, proj_vecs, points, ors)

        ang.nlist
        ang.projections
        ang.normed_projections
        ang.num_points
        ang.num_points
        ang.num_proj_vectors
        ang.box

    def test_compute(self):
        boxlen = 4
        num_neigh = 1
        rmax = 2

        box = freud.box.Box.cube(boxlen)

        proj_vecs = np.asarray([[0, 0, 1]])

        # Create three points in an L-shape.
        points = [[0, 0, 0]]
        points.append([1, 0, 0])
        points.append([0, 0, 1.5])
        # Three orientations:
        # 1. The identity
        ors = [[1, 0, 0, 0]]
        # 2. A rotation about the y axis by pi/2
        ors.append([np.cos(np.pi/4), 0, np.sin(np.pi/4), 0])
        # 3. A rotation about the z axis by pi/2
        ors.append([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])

        ors = np.asarray(ors, dtype=np.float32)
        points = np.asarray(points, dtype=np.float32)

        # First have no particle symmetry

        ang = freud.environment.LocalBondProjection(rmax, num_neigh)
        ang.compute(box, proj_vecs, points, ors)

        # For the first particle, the bond is [1,0,0] and the orientation is
        # [1,0,0,0]. So the local bond is [1,0,0]. The projection onto
        # [0,0,1] is cos(pi/2)=0.
        # In the new neighbor list, the corresponding bond is at index 2
        npt.assert_allclose(ang.projections[1], 0, atol=1e-6)
        npt.assert_allclose(ang.normed_projections[1], 0, atol=1e-6)
        # For the second particle, the bond is [-1,0,0] and the orientation is
        # rotated about the y-axis by pi/2. So the local bond is [0,0,-1]. The
        # projection onto [0,0,1] is cos(pi)=-1.
        # In the new neighbor list, the corresponding bond is at index 0
        npt.assert_allclose(ang.projections[0], -1, atol=1e-6)
        npt.assert_allclose(ang.normed_projections[0], -1, atol=1e-6)
        # For the third particle, the bond is [0,0,-1.5] and the orientation is
        # rotated about the z-axis by pi/2. So the local bond is [0,0,-1.5].
        # The projection onto [0,0,1] is 1.5*cos(pi)=-1.5
        # In the new neighbor list, the corresponding bond is at index 1
        # and the bond becomes [0, 0, 1.5]
        npt.assert_allclose(ang.projections[2], 1.5, atol=1e-6)
        npt.assert_allclose(ang.normed_projections[2], 1, atol=1e-6)

        # Specify that rotations about y by +/-pi/2 and rotations about x by pi
        # result in equivalent particle shapes
        qs = [[1, 0, 0, 0],
              [np.cos(np.pi/4), 0, np.sin(np.pi/4), 0],
              [np.cos(np.pi/2), np.sin(np.pi/2), 0, 0]]

        equiv_quats = []
        for q in qs:
            equiv_quats.append(q)
            # we have to include the adjoint (inverse) because this is a group
            equiv_quats.append(np.array([q[0], -q[1], -q[2], -q[3]]))
        equiv_quats = np.asarray(equiv_quats, dtype=np.float32)

        ang.compute(box, proj_vecs, points, ors, None, equiv_quats)

        # Now all projections should be cos(0)=1
        npt.assert_allclose(ang.projections[1], 1, atol=1e-6)
        npt.assert_allclose(ang.normed_projections[1], 1, atol=1e-6)
        npt.assert_allclose(ang.projections[0], 1, atol=1e-6)
        npt.assert_allclose(ang.normed_projections[0], 1, atol=1e-6)
        npt.assert_allclose(ang.projections[2], 1.5, atol=1e-6)
        npt.assert_allclose(ang.normed_projections[2], 1, atol=1e-6)

    def test_repr(self):
        ang = freud.environment.LocalBondProjection(3.0, 8)
        self.assertEqual(str(ang), str(eval(repr(ang))))


if __name__ == '__main__':
    unittest.main()
