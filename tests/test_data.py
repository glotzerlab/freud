import freud
import unittest
import numpy.testing as npt
import numpy as np


class TestData(unittest.TestCase):
    def test_square(self):
        """Test that the square lattice is correctly generated."""
        box, points = freud.data.UnitCell.square().to_system()
        self.assertEqual(box, freud.box.Box.square(1))
        npt.assert_array_equal(points, [[0, 0, 0]])

    def test_sc(self):
        """Test that the sc lattice is correctly generated."""
        box, points = freud.data.UnitCell.sc().to_system()
        self.assertEqual(box, freud.box.Box.cube(1))
        npt.assert_array_equal(points, [[0, 0, 0]])

    def test_bcc(self):
        """Test that the bcc lattice is correctly generated."""
        box, points = freud.data.UnitCell.bcc().to_system()
        self.assertEqual(box, freud.box.Box.cube(1))
        # Add a box.wrap to make sure there's no issues comparing periodic
        # images (e.g. 0.5 vs -0.5).
        npt.assert_array_equal(points, box.wrap([[.5, .5, .5], [0, 0, 0]]))

    def test_fcc(self):
        """Test that the fcc lattice is correctly generated."""
        box, points = freud.data.UnitCell.fcc().to_system()
        self.assertEqual(box, freud.box.Box.cube(1))
        # Add a box.wrap to make sure there's no issues comparing periodic
        # images (e.g. 0.5 vs -0.5).
        npt.assert_array_equal(points,
                               box.wrap([[.5, .5, 0],
                                         [.5, 0, .5],
                                         [0, .5, .5],
                                         [0, 0, 0]]))

    def test_scale(self):
        """Test the generation of a scaled structure."""
        for scale in [0.5, 2]:
            box, points = freud.data.UnitCell.fcc().to_system(scale=scale)
            self.assertEqual(box, freud.box.Box.cube(scale))
            npt.assert_array_equal(
                points,
                box.wrap(scale*np.array([[.5, .5, 0],
                                        [.5, 0, .5],
                                        [0, .5, .5],
                                        [0, 0, 0]])))

    def test_replicas(self):
        """Test that replication works."""
        num_replicas = 2
        box, points = freud.data.UnitCell.fcc().to_system(
            num_replicas=num_replicas)
        self.assertEqual(box, freud.box.Box.cube(num_replicas))

        test_points = np.array([[.5, .5, 0],
                                [.5, 0, .5],
                                [0, .5, .5],
                                [0, 0, 0]])
        test_points = test_points[np.newaxis, np.newaxis, np.newaxis, ...]
        test_points = np.tile(test_points,
                              [num_replicas, num_replicas, num_replicas, 1, 1])
        test_points[..., 0] += np.arange(
            num_replicas)[:, np.newaxis, np.newaxis, np.newaxis]
        test_points[..., 1] += np.arange(
            num_replicas)[np.newaxis, :, np.newaxis, np.newaxis]
        test_points[..., 2] += np.arange(
            num_replicas)[np.newaxis, np.newaxis, :, np.newaxis]
        test_points = (test_points-(0.5*num_replicas)).reshape(-1, 3)

        # Can't guarantee identical ordering based on algorithms.
        npt.assert_array_equal(
            set([tuple(x) for x in points]),
            set([tuple(x) for x in box.wrap(test_points)]),
        )


if __name__ == '__main__':
    unittest.main()
