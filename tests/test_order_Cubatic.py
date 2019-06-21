import numpy as np
import numpy.testing as npt
import freud
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


class TestCubatic(unittest.TestCase):
    def test_ordered(self):
        # do not need positions, just orientations
        N = 1000
        axes = np.zeros(shape=(N, 3), dtype=np.float32)
        angles = np.zeros(shape=N, dtype=np.float32)
        axes[:, 2] = 1.0

        # generate similar angles
        np.random.seed(1030)
        angles = np.random.uniform(low=0.0, high=0.05, size=N)

        # generate quaternions
        orientations = gen_quaternions(N, axes, angles)

        # create cubatic object
        t_initial = 5.0
        t_final = 0.001
        scale = 0.95
        n_replicates = 10
        cop = freud.order.CubaticOrderParameter(
            t_initial, t_final, scale, n_replicates)

        # Test access
        with self.assertRaises(AttributeError):
            cop.cubatic_order_parameter
        with self.assertRaises(AttributeError):
            cop.orientation
        with self.assertRaises(AttributeError):
            cop.particle_order_parameter
        with self.assertRaises(AttributeError):
            cop.particle_tensor
        with self.assertRaises(AttributeError):
            cop.global_tensor
        with self.assertRaises(AttributeError):
            cop.cubatic_tensor
        with self.assertRaises(AttributeError):
            cop.gen_r4_tensor

        cop.compute(orientations)

        # Test access
        cop.cubatic_order_parameter
        cop.orientation
        cop.particle_order_parameter
        cop.particle_tensor
        cop.global_tensor
        cop.cubatic_tensor
        cop.gen_r4_tensor

        # Test values of the OP
        self.assertAlmostEqual(cop.cubatic_order_parameter, 1, places=2,
                               msg="Cubatic Order is not approx. 1")
        self.assertGreater(np.nanmin(cop.particle_order_parameter), 0.9,
                           msg="Per particle order parameter value is too low")

        # Test attributes
        self.assertAlmostEqual(cop.t_initial, t_initial)
        self.assertAlmostEqual(cop.t_final, t_final)
        self.assertAlmostEqual(cop.scale, scale)

        # Test shapes for the tensor since we can't ensure values.
        self.assertEqual(cop.orientation.shape, (4,))
        self.assertEqual(
            cop.particle_tensor.shape, (len(orientations), 3, 3, 3, 3))
        self.assertEqual(cop.cubatic_tensor.shape, (3, 3, 3, 3))
        self.assertEqual(cop.global_tensor.shape, (3, 3, 3, 3))
        self.assertEqual(cop.gen_r4_tensor.shape, (3, 3, 3, 3))

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
        cubaticOP = freud.order.CubaticOrderParameter(5.0, 0.001, 0.95, 10)
        cubaticOP.compute(orientations)
        # get the op
        op = cubaticOP.cubatic_order_parameter

        pop = cubaticOP.particle_order_parameter
        op_max = np.nanmax(pop)

        npt.assert_array_less(op, 0.3, err_msg="Cubatic Order is > 0.3")
        npt.assert_array_less(
            op_max, 0.2,
            err_msg="per particle order parameter value is too high")

    def test_repr(self):
        cubaticOP = freud.order.CubaticOrderParameter(5.0, 0.001, 0.95, 10)
        self.assertEqual(str(cubaticOP), str(eval(repr(cubaticOP))))


if __name__ == '__main__':
    unittest.main()
