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

        ors = np.asarray(ors, dtype=np.float32)
        equiv_quats = np.asarray([[1,0,0,0]], dtype=np.float32)

        ang = freud.order.AngularSeparation(rmax, num_neigh)
        ang.computeNeighbor(box, ors, ors, points, points, equiv_quats)
        npt.assert_equal(ang.getNP(), N)

    def test_getNGlobal(self):
        N = 500
        num_neigh = 8
        rmax = 3

        ors = []
        for i in range(N):
            ors.append(quatRandom())

        ors = np.asarray(ors, dtype=np.float32)
        equiv_quats = np.asarray([[1,0,0,0]], dtype=np.float32)
        global_ors = np.array([[1,0,0,0]], dtype=np.float32)

        ang = freud.order.AngularSeparation(rmax, num_neigh)
        ang.computeGlobal(global_ors, ors, equiv_quats)
        npt.assert_equal(ang.getNGlobal(), 1)

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

        ors = np.asarray(ors, dtype=np.float32)
        equiv_quats = np.asarray([[1,0,0,0]], dtype=np.float32)

        ang = freud.order.AngularSeparation(rmax, num_neigh)
        ang.computeNeighbor(box, ors, ors, points, points, equiv_quats)
        npt.assert_equal(ang.getNReference(), N)


    def test_compute_neighbors(self):
        boxlen = 4
        N = 3
        num_neigh = 1
        rmax = 2
        
        box = freud.box.Box.square(boxlen)

        #Create three points in a line.
        points = [[0, 0, 0]]
        points.append([1, 0, 0])
        points.append([1.5, 0, 0])
        #Use two separate orientations. The second orientation is a pi/3 rotation from the identity quaternion
        ors = [[1, 0, 0, 0]]
        ors.append([np.cos(np.pi/6), np.sin(np.pi/6), 0, 0])
        ors.append([np.cos(np.pi/6), np.sin(np.pi/6), 0, 0])

        ors = np.asarray(ors, dtype=np.float32)
        points = np.asarray(points, dtype=np.float32)
        equiv_quats = np.asarray([[1,0,0,0], [-1, 0, 0, 0]], dtype=np.float32)

        ang = freud.order.AngularSeparation(rmax, num_neigh)
        ang.computeNeighbor(box, ors, ors, points, points, equiv_quats)
        
        #Should find that the angular separation between the first particle and its neighbor is pi/3
        #The second particle's nearest neighbor will have the same orientation
        npt.assert_almost_equal(ang.getNeighborAngles()[0], np.pi/3, 6)
        npt.assert_almost_equal(ang.getNeighborAngles()[1], 0, 6)

    def test_compute_global(self):
        N = 4 
        num_neigh = 1
        rmax = 2

        #Going to make sure that the use of equiv_quats captures both of the global reference orientations
        global_ors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        equiv_quats = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, -1, 0, 0]], dtype=np.float32)

        ors = [[1, 0, 0, 0]]
        ors.append([0, 1, 0, 0])
        #The following two quaternions correspond to rotations of the above by pi/16
        ors.append([ 0.99518473, 0., 0., 0.09801714])
        ors.append([ 0., 0.99518473, -0.09801714, 0.])

        ors = np.asarray(ors, dtype=np.float32)

        ang = freud.order.AngularSeparation(rmax, num_neigh)
        ang.computeGlobal(global_ors, ors, equiv_quats)

        #Each orientation should be either equal to or pi/16 away from the global reference quaternion
        for i in [0, 1]:
            for j in [0, 1]:
                npt.assert_almost_equal(ang.getGlobalAngles()[i][j], 0, 6)
        for i in [2, 3]:
            for j in [0, 1]:
                npt.assert_almost_equal(ang.getGlobalAngles()[i][j], np.pi/16, 6)


if __name__ == '__main__':
    unittest.main()
