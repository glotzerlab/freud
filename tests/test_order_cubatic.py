import numpy as np
import numpy.testing as npt
from freud.order import CubaticOrderParameter as cop
from freud import trajectory
import unittest

def gen_quaternions(n, axes, angles):
    q = np.zeros(shape=(n, 4), dtype=np.float32)
    for i, (axis, angle) in enumerate(zip(axes, angles)):
        q[i] = [np.cos(angle/2.0), np.sin(angle/2.0) * axis[0], np.sin(angle/2.0) * axis[1], np.sin(angle/2.0) * axis[2]]
        q[i] /= np.linalg.norm(q[i])
    return q

class TestCluster(unittest.TestCase):
    def test_ordered(self):
        # do not need positions, just orientations
        N = 100
        axes = np.zeros(shape=(N, 3), dtype=np.float32)
        angles = np.zeros(shape=N, dtype=np.float32)
        axes[:,2] = 1.0

        # generate similar angles
        angles = np.random.uniform(low=0.0, high=0.05, size=N)

        # generate quaternions
        orientations = gen_quaternions(N, axes, angles)

        # create cubatic object
        cubaticOP = cop(5.0, 0.001, 0.95, 10)
        cubaticOP.compute(orientations)
        # get the op
        op = cubaticOP.get_cubatic_order_parameter()

        npt.assert_almost_equal(op, 1, decimal=2, err_msg="Cubatic Order is not apprx 1")

    def test_disordered(self):
        # do not need positions, just orientations
        N = 100
        axes = np.zeros(shape=(N, 3), dtype=np.float32)
        angles = np.zeros(shape=N, dtype=np.float32)
        # pick axis at random
        for i in range(N):
            axis = np.random.randint(low=0, high=3)
            axes[:,axis] = 1.0

        # generate similar angles
        angles = np.random.uniform(low=0.0, high=2.0*np.pi, size=N)

        # generate quaternions
        orientations = gen_quaternions(N, axes, angles)

        # create cubatic object
        cubaticOP = cop(5.0, 0.001, 0.95, 10)
        cubaticOP.compute(orientations)
        # get the op
        op = cubaticOP.get_cubatic_order_parameter()

        npt.assert_array_less(op, 0.5, err_msg="Cubatic Order is > 0.5")


if __name__ == '__main__':
    unittest.main()

