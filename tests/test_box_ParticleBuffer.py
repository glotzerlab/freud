import numpy as np
import numpy.testing as npt
import freud
import unittest


class TestParticleBuffer(unittest.TestCase):
    def test_square(self):
        L = 10          # Box length
        N = 50          # Number of particles
        np.random.seed(0)

        fbox = freud.box.Box.square(L)  # Initialize box
        pbuff = freud.box.ParticleBuffer(fbox)

        # Generate random points in the box
        positions = np.random.uniform(-L/2, L/2, size=(N, 2))

        # Add a z-component of 0
        positions = np.insert(positions, 2, 0, axis=1).astype(np.float32)

        # Compute with zero buffer distance
        pbuff.compute(positions, buffer=0, images=False)
        self.assertEqual(len(pbuff.buffer_particles), 0)
        npt.assert_array_equal(pbuff.buffer_box.L, np.asarray(fbox.L))

        # Compute with buffer distances
        pbuff.compute(positions, buffer=0.5*L, images=False)
        self.assertEqual(len(pbuff.buffer_particles), 3 * N)
        npt.assert_array_equal(pbuff.buffer_box.L, 2 * np.asarray(fbox.L))

        # Compute with zero images
        pbuff.compute(positions, buffer=0, images=True)
        self.assertEqual(len(pbuff.buffer_particles), 0)
        npt.assert_array_equal(pbuff.buffer_box.L, np.asarray(fbox.L))

        # Compute with images
        pbuff.compute(positions, buffer=1, images=True)
        self.assertEqual(len(pbuff.buffer_particles), 3 * N)
        npt.assert_array_equal(pbuff.buffer_box.L, 2 * np.asarray(fbox.L))

    def test_cube(self):
        L = 10  # Box length
        N = 50  # Number of particles
        np.random.seed(0)

        fbox = freud.box.Box.cube(L)  # Initialize box
        pbuff = freud.box.ParticleBuffer(fbox)

        # Generate random points in the box
        positions = np.random.uniform(-L/2, L/2, size=(N, 3))

        # Compute with zero buffer distance
        pbuff.compute(positions, buffer=0, images=False)
        self.assertEqual(len(pbuff.buffer_particles), 0)
        npt.assert_array_equal(pbuff.buffer_box.L, np.asarray(fbox.L))

        # Compute with buffer distances
        pbuff.compute(positions, buffer=0.5*L, images=False)
        self.assertEqual(len(pbuff.buffer_particles), 7 * N)
        npt.assert_array_equal(pbuff.buffer_box.L, 2 * np.asarray(fbox.L))

        # Compute with zero images
        pbuff.compute(positions, buffer=0, images=True)
        self.assertEqual(len(pbuff.buffer_particles), 0)
        npt.assert_array_equal(pbuff.buffer_box.L, np.asarray(fbox.L))

        # Compute with images
        pbuff.compute(positions, buffer=1, images=True)
        self.assertEqual(len(pbuff.buffer_particles), 7 * N)
        npt.assert_array_equal(pbuff.buffer_box.L, 2 * np.asarray(fbox.L))

    def test_triclinic(self):
        N = 50  # Number of particles
        np.random.seed(0)

        fbox = freud.box.Box(Lx=2, Ly=2, Lz=2, xy=1, xz=0, yz=1)
        pbuff = freud.box.ParticleBuffer(fbox)

        # Generate random points in the box, in fractional coordinates
        positions = np.random.uniform(0, 1, size=(N, 3))

        # Convert fractional coordinates to real coordinates
        positions = np.asarray(list(map(fbox.makeCoordinates, positions)))
        positions = fbox.wrap(positions)

        # Compute with zero images
        pbuff.compute(positions, buffer=0, images=True)
        self.assertEqual(len(pbuff.buffer_particles), 0)
        npt.assert_array_equal(pbuff.buffer_box.L, np.asarray(fbox.L))

        # Compute with images
        pbuff.compute(positions, buffer=2, images=True)
        self.assertEqual(len(pbuff.buffer_particles), 26 * N)
        npt.assert_array_equal(pbuff.buffer_box.L, 3 * np.asarray(fbox.L))


if __name__ == '__main__':
    unittest.main()
