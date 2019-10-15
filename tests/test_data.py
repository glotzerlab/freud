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
        self.assertEqual(
            set([tuple(x) for x in points]),
            set([tuple(x) for x in box.wrap(test_points)]),
        )

    def test_noise(self):
        """Test that noise generation works."""
        sigma = 0.01
        box, points = freud.data.UnitCell.fcc().to_system(
            sigma_noise=sigma, seed=0)
        self.assertEqual(box, freud.box.Box.cube(1))

        test_points = box.wrap(np.array([[.5, .5, 0],
                                         [.5, 0, .5],
                                         [0, .5, .5],
                                         [0, 0, 0]]))

        deltas = np.linalg.norm(box.wrap(test_points-points), axis=-1)
        # Nothing should be exactly equal, but differences should not be too
        # large. 4 sigma is an arbitrary choice that gives a high probability
        # of the test passing (although not necessary since the seed is set
        # above).
        self.assertFalse(np.allclose(deltas, 0))
        npt.assert_allclose(deltas, 0, atol=4*sigma)

    def test_seed(self):
        """Ensure that seeding does not overwrite the global state."""
        num_points = 10
        sigma = 0.01

        np.random.seed(0)
        first_rand = np.random.randint(num_points)
        box, points = freud.data.UnitCell.fcc().to_system(
            sigma_noise=sigma, seed=1)
        second_rand = np.random.randint(num_points)

        np.random.seed(0)
        third_rand = np.random.randint(num_points)
        box, points = freud.data.UnitCell.fcc().to_system(
            sigma_noise=sigma, seed=2)
        fourth_rand = np.random.randint(num_points)

        npt.assert_array_equal(first_rand, third_rand)
        npt.assert_array_equal(second_rand, fourth_rand)


if __name__ == '__main__':
    unittest.main()
