import numpy as np
from freud.order import NematicOrderParameter as nop
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


class TestNematicOrder(unittest.TestCase):
    def test_perfect(self):
        """Test perfectly aligned systems with different molecular axes"""
        N = 1000
        axes = np.zeros(shape=(N, 3), dtype=np.float32)
        angles = np.zeros(shape=N, dtype=np.float32)
        axes[:, 0] = 1.0
        orientations = gen_quaternions(N, axes, angles)

        # Test for parallel to molecular axis
        u = np.array([1, 0, 0])
        op_parallel = nop(u)
        op_parallel.compute(orientations)

        self.assertTrue(op_parallel.get_nematic_order_parameter() == 1)
        self.assertTrue(np.all(op_parallel.get_director() == u))
        self.assertTrue(np.all(
            op_parallel.get_nematic_tensor() == np.diag([1, -0.5, -0.5])))

        # Test for perpendicular to molecular axis
        u = np.array([0, 1, 0])
        op_perp = nop(u)
        op_perp.compute(orientations)

        self.assertTrue(op_perp.get_nematic_order_parameter() == 1)
        self.assertTrue(np.all(op_perp.get_director() == u))
        self.assertTrue(np.all(
            op_perp.get_nematic_tensor() == np.diag([-0.5, 1, -0.5])))

    def test_imperfect(self):
        """Test imperfectly aligned systems.
        Note that since two evals for the tests below, there are two possible
        choices of director. This test is currently assuming that the internal
        logic for choosing which one is the director does not change.
        """
        N = 1000
        axes = np.zeros(shape=(N, 3), dtype=np.float32)
        angles = np.zeros(shape=N, dtype=np.float32)
        # Rotating 90 about y gives some tensor components in z
        axes[::2, 0] = 1.0
        axes[1::2, 1] = 1.0
        angles[:] = np.pi/2
        orientations = gen_quaternions(N, axes, angles)

        u = np.array([1, 0, 0])
        op = nop(u)
        op.compute(orientations)

        self.assertTrue(np.allclose(op.get_nematic_order_parameter(), 0.25))
        self.assertTrue(np.all(op.get_director() == u))
        self.assertTrue(np.allclose(
            op.get_nematic_tensor(), np.diag([0.25, -0.5, 0.25])))

        # Rotating 90 about z gives some tensor components in y
        axes = np.zeros(shape=(N, 3), dtype=np.float32)
        angles = np.zeros(shape=N, dtype=np.float32)
        axes[1::2, 1] = 0.0
        axes[1::2, 2] = 1.0
        angles[:] = np.pi/2
        orientations = gen_quaternions(N, axes, angles)

        u = np.array([1, 0, 0])
        op = nop(u)
        op.compute(orientations)

        self.assertTrue(np.allclose(op.get_nematic_order_parameter(), 0.25))
        self.assertTrue(np.all(op.get_director() == np.array([0, 1, 0])))
        self.assertTrue(np.allclose(
            op.get_nematic_tensor(), np.diag([0.25, 0.25, -0.5])))


if __name__ == '__main__':
    unittest.main()
