import numpy as np
import numpy.testing as npt
import freud
from freud.order import LocalDescriptors
import unittest

class TestLocalDescriptors(unittest.TestCase):
    def test_shape(self):
        N = 1000
        Nneigh = 4
        lmax = 8

        box = freud.box.Box.cube(10)
        positions = np.random.uniform(-box.getLx()/2, box.getLx()/2, size=(N, 3)).astype(np.float32)

        comp = LocalDescriptors(Nneigh, lmax, .5, True)
        comp.computeNList(box, positions)
        comp.compute(box, Nneigh, positions)

        sphs = comp.getSph()

        assert sphs.shape[0] == N*Nneigh

    def test_global(self):
        N = 1000
        Nneigh = 4
        lmax = 8

        box = freud.box.Box.cube(10)
        positions = np.random.uniform(-box.getLx()/2, box.getLx()/2, size=(N, 3)).astype(np.float32)

        comp = LocalDescriptors(Nneigh, lmax, .5, True)
        comp.computeNList(box, positions)
        comp.compute(box, Nneigh, positions, mode='global')

        sphs = comp.getSph()

        assert sphs.shape[0] == N*Nneigh

    def test_particle_local(self):
        N = 1000
        Nneigh = 4
        lmax = 8

        box = freud.box.Box.cube(10)
        positions = np.random.uniform(-box.getLx()/2, box.getLx()/2, size=(N, 3)).astype(np.float32)
        orientations = np.random.uniform(-1, 1, size=(N, 4)).astype(np.float32)
        orientations /= np.sqrt(np.sum(orientations**2, axis=-1))[:, np.newaxis]

        comp = LocalDescriptors(Nneigh, lmax, .5, True)
        comp.computeNList(box, positions)

        with self.assertRaises(RuntimeError):
            comp.compute(box, Nneigh, positions, mode='particle_local')

        comp.compute(box, Nneigh, positions, orientations=orientations, mode='particle_local')

        sphs = comp.getSph()

        assert sphs.shape[0] == N*Nneigh

    def test_unknown_modes(self):
        N = 1000
        Nneigh = 4
        lmax = 8

        box = freud.box.Box.cube(10)
        positions = np.random.uniform(-box.getLx()/2, box.getLx()/2, size=(N, 3)).astype(np.float32)

        comp = LocalDescriptors(Nneigh, lmax, .5, True)
        comp.computeNList(box, positions)

        with self.assertRaises(RuntimeError):
            comp.compute(box, Nneigh, positions, mode='particle_local_wrong')

    def test_shape_twosets(self):
        N = 1000
        Nneigh = 4
        lmax = 8

        box = freud.box.Box.cube(10)
        positions = np.random.uniform(-box.getLx()/2, box.getLx()/2, size=(N, 3)).astype(np.float32)
        positions2 = np.random.uniform(-box.getLx()/2, box.getLx()/2, size=(N//3, 3)).astype(np.float32)

        comp = LocalDescriptors(Nneigh, lmax, .5, True)
        comp.computeNList(box, positions, positions2)
        comp.compute(box, Nneigh, positions, positions2)

        lc = freud.locality.LinkCell(box, 2).compute(box, positions, positions2)
        nn = freud.locality.NearestNeighbors(1, 4).compute(box, positions, positions2)

        sphs = comp.getSph()

        assert sphs.shape[0] == N*Nneigh

if __name__ == '__main__':
    unittest.main()
