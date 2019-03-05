import numpy as np
import freud
from freud.environment import LocalDescriptors
import unittest


class TestLocalDescriptors(unittest.TestCase):
    def test_shape(self):
        N = 1000
        Nneigh = 4
        lmax = 8
        rmax = 0.5

        box = freud.box.Box.cube(10)
        np.random.seed(0)
        positions = np.random.uniform(-box.Lx/2, box.Lx/2,
                                      size=(N, 3)).astype(np.float32)

        comp = LocalDescriptors(Nneigh, lmax, rmax, True)
        comp.computeNList(box, positions)
        comp.compute(box, Nneigh, positions)

        self.assertEqual(comp.sph.shape[0], N*Nneigh)

        self.assertEqual(comp.num_particles, positions.shape[0])

        self.assertEqual(comp.num_neighbors/comp.num_particles, Nneigh)

        self.assertEqual(comp.l_max, lmax)

    def test_global(self):
        N = 1000
        Nneigh = 4
        lmax = 8

        box = freud.box.Box.cube(10)
        np.random.seed(0)
        positions = np.random.uniform(-box.Lx/2, box.Lx/2,
                                      size=(N, 3)).astype(np.float32)

        comp = LocalDescriptors(Nneigh, lmax, .5, True)
        comp.computeNList(box, positions)
        comp.compute(box, Nneigh, positions, mode='global')

        sphs = comp.sph

        self.assertEqual(sphs.shape[0], N*Nneigh)

    def test_particle_local(self):
        N = 1000
        Nneigh = 4
        lmax = 8

        box = freud.box.Box.cube(10)
        np.random.seed(0)
        positions = np.random.uniform(-box.Lx/2, box.Lx/2,
                                      size=(N, 3)).astype(np.float32)
        orientations = np.random.uniform(-1, 1, size=(N, 4)).astype(np.float32)
        orientations /= np.sqrt(np.sum(orientations**2,
                                       axis=-1))[:, np.newaxis]

        comp = LocalDescriptors(Nneigh, lmax, .5, True)
        comp.computeNList(box, positions)

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

        box = freud.box.Box.cube(10)
        np.random.seed(0)
        positions = np.random.uniform(-box.Lx/2, box.Lx/2,
                                      size=(N, 3)).astype(np.float32)

        comp = LocalDescriptors(Nneigh, lmax, .5, True)
        comp.computeNList(box, positions)

        with self.assertRaises(RuntimeError):
            comp.compute(box, Nneigh, positions, mode='particle_local_wrong')

    def test_shape_twosets(self):
        N = 1000
        Nneigh = 4
        lmax = 8

        box = freud.box.Box.cube(10)
        np.random.seed(0)
        positions = np.random.uniform(-box.Lx/2, box.Lx/2,
                                      size=(N, 3)).astype(np.float32)
        positions2 = np.random.uniform(-box.Lx/2, box.Lx/2,
                                       size=(N//3, 3)).astype(np.float32)

        comp = LocalDescriptors(Nneigh, lmax, .5, True)
        comp.computeNList(box, positions, positions2)
        comp.compute(box, Nneigh, positions, positions2)
        sphs = comp.sph
        self.assertEqual(sphs.shape[0], N*Nneigh)


if __name__ == '__main__':
    unittest.main()
