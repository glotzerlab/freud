import numpy as np
import numpy.testing as npt
import freud
import rowan
import unittest


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
        orientations = rowan.from_axis_angle(axes, angles)

        # create cubatic object
        t_initial = 5.0
        t_final = 0.001
        scale = 0.95
        n_replicates = 10
        cop = freud.order.Cubatic(
            t_initial, t_final, scale, n_replicates)

        # Test access
        with self.assertRaises(AttributeError):
            cop.order
        with self.assertRaises(AttributeError):
            cop.orientation
        with self.assertRaises(AttributeError):
            cop.particle_order
        with self.assertRaises(AttributeError):
            cop.global_tensor
        with self.assertRaises(AttributeError):
            cop.cubatic_tensor

        cop.compute(orientations)

        # Test access
        cop.order
        cop.orientation
        cop.particle_order
        cop.global_tensor
        cop.cubatic_tensor

        # Test values of the OP
        self.assertAlmostEqual(cop.order, 1, places=2,
                               msg="Cubatic Order is not approx. 1")
        self.assertGreater(np.nanmin(cop.particle_order), 0.9,
                           msg="Per particle order parameter value is too low")

        # Test attributes
        self.assertAlmostEqual(cop.t_initial, t_initial)
        self.assertAlmostEqual(cop.t_final, t_final)
        self.assertAlmostEqual(cop.scale, scale)

        # Test shapes for the tensor since we can't ensure values.
        self.assertEqual(cop.orientation.shape, (4,))
        self.assertEqual(cop.cubatic_tensor.shape, (3, 3, 3, 3))
        self.assertEqual(cop.global_tensor.shape, (3, 3, 3, 3))

    def test_disordered(self):
        # do not need positions, just orientations
        N = 1000
        axes = np.zeros(shape=(N, 3), dtype=np.float32)
        angles = np.zeros(shape=N, dtype=np.float32)
        # pick axis at random
        ax_list = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                            [1, 1, 0], [1, 0, 1], [0, 1, 1],
                            [1, 1, 1]], dtype=np.float32)
        ax_list /= np.linalg.norm(ax_list, axis=-1)[:, np.newaxis]

        for i in range(N):
            axes[i] = ax_list[i % ax_list.shape[0]]

        # generate disordered orientations
        np.random.seed(0)
        angles = np.random.uniform(low=np.pi/4.0, high=np.pi/2.0, size=N)
        orientations = rowan.from_axis_angle(axes, angles)

        # create cubatic object
        cubatic = freud.order.Cubatic(5.0, 0.001, 0.95, 10)
        cubatic.compute(orientations)
        # get the op
        op = cubatic.order

        pop = cubatic.particle_order
        op_max = np.nanmax(pop)

        npt.assert_array_less(op, 0.3, err_msg="Cubatic Order is > 0.3")
        npt.assert_array_less(
            op_max, 0.2,
            err_msg="per particle order parameter value is too high")

    def test_valid_inputs(self):
        with self.assertRaises(ValueError):
            # t_initial must be greater than t_final
            freud.order.Cubatic(
                t_initial=0.001,
                t_final=5.0,
                scale=0.95,
                n_replicates=10)

        with self.assertRaises(ValueError):
            # t_final must be greater than 1e-6
            freud.order.Cubatic(
                t_initial=5.0,
                t_final=1e-7,
                scale=0.95,
                n_replicates=10)

        with self.assertRaises(ValueError):
            # scale must be less than 1
            freud.order.Cubatic(
                t_initial=5.0,
                t_final=0.001,
                scale=1,
                n_replicates=10)

        with self.assertRaises(ValueError):
            # scale must be greater than 0
            freud.order.Cubatic(
                t_initial=5.0,
                t_final=0.001,
                scale=0,
                n_replicates=10)

    def test_repr(self):
        cubatic = freud.order.Cubatic(5.0, 0.001, 0.95, 10)
        self.assertEqual(str(cubatic), str(eval(repr(cubatic))))


if __name__ == '__main__':
    unittest.main()
