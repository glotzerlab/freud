import unittest
import numpy.testing as npt
import numpy as np
import random
import freud

## Returns a random quaternion culled from a uniform distribution on the surface of a 3-sphere.
## Uses the MARSAGLIA (1972) method (a la hoomd)
## NOTE THAT generating a random rotation via a random angle about a random axis of rotation is INCORRECT.
## See K. Shoemake, "Uniform Random Rotations," 1992, for a nice explanation for this.
# output quat is an array of four numbers: [q0, q1, q2, q3]
def quatRandom():
    # random.uniform(a,b) gives number in [a,b]
    v1 = random.uniform(-1,1)
    v2 = random.uniform(-1,1)
    v3 = random.uniform(-1,1)
    v4 = random.uniform(-1,1)

    s1 = v1*v1 + v2*v2
    s2 = v3*v3 + v4*v4

    while (s1 >= 1.):
        v1 = random.uniform(-1,1)
        v2 = random.uniform(-1,1)
        s1 = v1*v1 + v2*v2

    while (s2 >= 1. or s2 == 0.):
        v3 = random.uniform(-1,1)
        v4 = random.uniform(-1,1)
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

        points = np.asarray(np.random.uniform(-boxlen/2, boxlen/2, (N,3)), dtype=np.float32)
        ors = []
        for i in range(N):
            ors.append(quatRandom())

        ors = np.asarray(ors)
        equiv_quats = np.asarray([[1,0,0,0]])

        ang = freud.order.AngularSeparation(rmax, num_neigh)
        ang.computeNeighbor(box, ors, ors, points, points, equiv_quats)
        npt.assert_equal(hop.getNP(), N)

    def test_getNGlobal(self):
        N = 500

        ors = []
        for i in range(N):
            ors.append(quatRandom())

        ors = np.asarray(ors)
        equiv_quats = np.asarray([[1,0,0,0]])
        global_ors = np.array([[1,0,0,0]])

        ang = freud.order.AngularSeparation(rmax, num_neigh)
        ang.computeGlobal(global_ors, ors, equiv_quats)
        npt.assert_equal(hop.getNGlobal(), 1)

    def test_getNReference(self):
        boxlen = 10
        N = 500
        num_neigh = 8
        rmax = 3

        box = freud.box.Box.square(boxlen)

        points = np.asarray(np.random.uniform(-boxlen/2, boxlen/2, (N,3)), dtype=np.float32)
        ors = []
        for i in range(N):
            ors.append(quatRandom())

        ors = np.asarray(ors)
        equiv_quats = np.asarray([[1,0,0,0]])

        ang = freud.order.AngularSeparation(rmax, num_neigh)
        ang.computeNeighbor(box, ors, ors, points, points, equiv_quats)
        npt.assert_equal(hop.getNReference(), N)

    def test_compute(self):
        boxlen = 10
        N = 7
        rmax = 3

        box = freud.box.Box.square(boxlen)

        points = [[0.0, 0.0, 0.0]]

        for i in range(6):
            points.append([np.cos(float(i) * 2.0 * np.pi / 6.0),
                           np.sin(float(i) * 2.0 * np.pi / 6.0),
                           0.0])

        points = np.asarray(points, dtype=np.float32)
        points[:,2] = 0.0
        hop = freud.order.HexOrderParameter(rmax)
        hop.compute(box, points)
        npt.assert_almost_equal(hop.getPsi()[0], 1. + 0.j, decimal=1)

if __name__ == '__main__':
    unittest.main()
