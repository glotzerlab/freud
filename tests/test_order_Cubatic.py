import numpy as np
import numpy.testing as npt
from freud.order import CubaticOrderParameter as cop
import unittest


def gen_quaternions(n, axes, angles):
    q = np.zeros(shape=(n, 4), dtype=np.float32)
    for i, (axis, angle) in enumerate(zip(axes, angles)):
        q[i] = [np.cos(angle/2.0),
                np.sin(angle/2.0) * axis[0],
                np.sin(angle/2.0) * axis[1],
                np.sin(angle/2.0) * axis[2]]
        q[i] /= np.linalg.norm(q[i])
    return q


class TestCluster(unittest.TestCase):
    def test_ordered(self):
        # do not need positions, just orientations
        N = 1000
        axes = np.zeros(shape=(N, 3), dtype=np.float32)
        angles = np.zeros(shape=N, dtype=np.float32)
        axes[:, 2] = 1.0

        # generate similar angles
        np.random.seed(0)
        angles = np.random.uniform(low=0.0, high=0.05, size=N)

        # generate quaternions
        orientations = gen_quaternions(N, axes, angles)

        # create cubatic object
        cubaticOP = cop(5.0, 0.001, 0.95, 10)
        cubaticOP.compute(orientations)

        # get the op
        op = cubaticOP.get_cubatic_order_parameter()

        # simple testing
        pop = cubaticOP.get_particle_op()
        op_min = np.nanmin(pop)

        npt.assert_almost_equal(op, 1, decimal=2,
                                err_msg="Cubatic Order is not approx. 1")
        npt.assert_array_less(
            0.9, op_min,
            err_msg="per particle order parameter value is too low")

    @unittest.skip("This test appears to be flawed, "
                   "for some random angles it can fail")
    def test_disordered(self):
        # do not need positions, just orientations
        N = 1000
        axes = np.zeros(shape=(N, 3), dtype=np.float32)
        angles = np.zeros(shape=N, dtype=np.float32)
        # pick axis at random
        ax_list = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                            [1, 1, 0], [1, 0, 1], [0, 1, 1],
                            [1, 1, 1]], dtype=np.float32)
        for ax in ax_list:
            ax /= np.linalg.norm(ax)
        for i in range(N):
            axes[i] = ax_list[i % ax_list.shape[0]]

        # generate disordered angles
        np.random.seed(0)
        angles = np.random.uniform(low=np.pi/4.0, high=np.pi/2.0, size=N)

        # generate quaternions
        orientations = gen_quaternions(N, axes, angles)

        # create cubatic object
        cubaticOP = cop(5.0, 0.001, 0.95, 10)
        cubaticOP.compute(orientations)
        # get the op
        op = cubaticOP.get_cubatic_order_parameter()

        pop = cubaticOP.get_particle_op()
        op_max = np.nanmax(pop)

        npt.assert_array_less(op, 0.3, err_msg="Cubatic Order is > 0.3")
        npt.assert_array_less(
            op_max, 0.2,
            err_msg="per particle order parameter value is too high")


if __name__ == '__main__':
    unittest.main()
