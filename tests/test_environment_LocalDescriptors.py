import numpy as np
import freud
import unittest
from util import makeBoxAndRandomPoints


class TestLocalDescriptors(unittest.TestCase):
    def test_shape(self):
        N = 1000
        Nneigh = 4
        lmax = 8
        rmax = 0.5
        L = 10

        box, positions = makeBoxAndRandomPoints(L, N)
        positions.flags['WRITEABLE'] = False

        comp = freud.environment.LocalDescriptors(Nneigh, lmax, rmax, True)

        # Test access
        with self.assertRaises(AttributeError):
            comp.sph
        with self.assertRaises(AttributeError):
            comp.num_particles
        with self.assertRaises(AttributeError):
            comp.num_neighbors

        comp.compute(box, Nneigh, positions)

        # Test access
        comp.sph
        comp.num_particles
        comp.num_neighbors

        self.assertEqual(comp.sph.shape[0], N*Nneigh)

        self.assertEqual(comp.num_particles, positions.shape[0])

        self.assertEqual(comp.num_neighbors/comp.num_particles, Nneigh)

        self.assertEqual(comp.l_max, lmax)

    def test_global(self):
        N = 1000
        Nneigh = 4
        lmax = 8
        L = 10

        box, positions = makeBoxAndRandomPoints(L, N)

        comp = freud.environment.LocalDescriptors(Nneigh, lmax, .5, True)
        comp.compute(box, Nneigh, positions, mode='global')

        sphs = comp.sph

        self.assertEqual(sphs.shape[0], N*Nneigh)

    def test_particle_local(self):
        N = 1000
        Nneigh = 4
        lmax = 8
        L = 10

        box, positions = makeBoxAndRandomPoints(L, N)
        orientations = np.random.uniform(-1, 1, size=(N, 4)).astype(np.float32)
        orientations /= np.sqrt(np.sum(orientations**2,
                                       axis=-1))[:, np.newaxis]

        comp = freud.environment.LocalDescriptors(Nneigh, lmax, .5, True)

        with self.assertRaises(RuntimeError):
            comp.compute(box, Nneigh, positions, mode='particle_local')

        comp.compute(box, Nneigh, positions,
                     orientations=orientations, mode='particle_local')

        sphs = comp.sph

        self.assertEqual(sphs.shape[0], N*Nneigh)

    def test_unknown_modes(self):
        N = 1000
        Nneigh = 4
        lmax = 8
        L = 10

        box, positions = makeBoxAndRandomPoints(L, N)

        comp = freud.environment.LocalDescriptors(Nneigh, lmax, .5, True)

        with self.assertRaises(RuntimeError):
            comp.compute(box, Nneigh, positions, mode='particle_local_wrong')

    def test_shape_twosets(self):
        N = 1000
        Nneigh = 4
        lmax = 8
        L = 10

        box, positions = makeBoxAndRandomPoints(L, N)
        positions2 = np.random.uniform(-L/2, L/2,
                                       size=(N//3, 3)).astype(np.float32)

        comp = freud.environment.LocalDescriptors(Nneigh, lmax, .5, True)
        comp.compute(box, Nneigh, positions, positions2)
        sphs = comp.sph
        self.assertEqual(sphs.shape[0], N//3*Nneigh)

    def test_repr(self):
        comp = freud.environment.LocalDescriptors(4, 8, 0.5, True)
        self.assertEqual(str(comp), str(eval(repr(comp))))


if __name__ == '__main__':
    unittest.main()
