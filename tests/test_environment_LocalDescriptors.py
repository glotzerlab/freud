import numpy as np
import numpy.testing as npt
import freud
import unittest
from util import make_box_and_random_points, make_sc, make_bcc, make_fcc


class TestLocalDescriptors(unittest.TestCase):
    def test_shape(self):
        N = 1000
        Nneigh = 4
        lmax = 8
        rmax = 0.5
        L = 10

        box, positions = make_box_and_random_points(L, N)
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

        box, positions = make_box_and_random_points(L, N)

        comp = freud.environment.LocalDescriptors(Nneigh, lmax, .5, True)
        comp.compute(box, Nneigh, positions, mode='global')

        sphs = comp.sph

        self.assertEqual(sphs.shape[0], N*Nneigh)

    def test_particle_local(self):
        N = 1000
        Nneigh = 4
        lmax = 8
        L = 10

        box, positions = make_box_and_random_points(L, N)
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

        box, positions = make_box_and_random_points(L, N)

        comp = freud.environment.LocalDescriptors(Nneigh, lmax, .5, True)

        with self.assertRaises(RuntimeError):
            comp.compute(box, Nneigh, positions, mode='particle_local_wrong')

    def test_shape_twosets(self):
        N = 1000
        Nneigh = 4
        lmax = 8
        L = 10

        box, positions = make_box_and_random_points(L, N)
        positions2 = np.random.uniform(-L/2, L/2,
                                       size=(N//3, 3)).astype(np.float32)

        comp = freud.environment.LocalDescriptors(Nneigh, lmax, .5, True)
        comp.compute(box, Nneigh, positions, positions2)
        sphs = comp.sph
        self.assertEqual(sphs.shape[0], N*Nneigh)

    def test_repr(self):
        comp = freud.environment.LocalDescriptors(4, 8, 0.5, True)
        self.assertEqual(str(comp), str(eval(repr(comp))))

    def test_ql(self):
        """Check if we can reproduce Steinhardt OPs."""
        def get_Ql(p, descriptors, nlist):
            """Given a set of points and a LocalDescriptors object (and the
            underlying neighborlist, compute the per-particle Steinhardt order
            parameter for all :math:`l` values up to the maximum quantum number
            used in the computation of the descriptors."""
            Qbar_lm = np.zeros((p.shape[0], descriptors.sph.shape[1]),
                               dtype=np.complex128)
            num_neighbors = descriptors.sph.shape[0]/p.shape[0]
            for i in range(p.shape[0]):
                indices = nlist.index_i == i
                Qbar_lm[i, :] = np.sum(descriptors.sph[indices, :],
                                       axis=0)/num_neighbors

            Ql = np.zeros((Qbar_lm.shape[0], descriptors.l_max+1))
            for i in range(Ql.shape[0]):
                for l in range(Ql.shape[1]):
                    for k in range(l**2, (l+1)**2):
                        Ql[i, l] += np.absolute(Qbar_lm[i, k])**2
                    Ql[i, l] = np.sqrt(4*np.pi/(2*l + 1) * Ql[i, l])

            return Ql

        # These exact parameter values aren't important; they won't necessarily
        # give useful outputs for some of the structures, but that's fine since
        # we just want to check that LocalDescriptors is consistent with
        # Steinhardt.
        num_neighbors = 6
        l_max = 12
        r_max = 2

        for struct_func in [make_sc, make_bcc, make_fcc]:
            box, points = struct_func(5, 5, 5)

            # In order to be able to access information on which particles are
            # bonded to which ones, we precompute the neighborlist
            nn = freud.locality.NearestNeighbors(r_max, num_neighbors)
            nl = nn.compute(box, points).nlist
            ld = freud.environment.LocalDescriptors(
                num_neighbors, l_max, r_max)
            ld.compute(box, num_neighbors, points, mode='global', nlist=nl)

            Ql = get_Ql(points, ld, nl)

            # Test all allowable values of l.
            for L in range(2, l_max+1):
                steinhardt = freud.order.Steinhardt(r_max*2, L)
                steinhardt.compute(box, points, nlist=nl)
                npt.assert_array_almost_equal(steinhardt.order, Ql[:, L])


if __name__ == '__main__':
    unittest.main()
