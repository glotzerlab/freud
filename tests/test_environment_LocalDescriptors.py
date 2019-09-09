import numpy as np
import numpy.testing as npt
import freud
import sys
import unittest
from util import (make_box_and_random_points, make_sc, make_bcc, make_fcc,
                  skipIfMissing)


class TestLocalDescriptors(unittest.TestCase):
    def test_shape(self):
        N = 1000
        num_neighbors = 4
        l_max = 8
        r_max = 0.5
        L = 10

        box, positions = make_box_and_random_points(L, N)
        positions.flags['WRITEABLE'] = False

        comp = freud.environment.LocalDescriptors(
            num_neighbors, l_max, r_max, True)

        # Test access
        with self.assertRaises(AttributeError):
            comp.sph
        with self.assertRaises(AttributeError):
            comp.num_particles
        with self.assertRaises(AttributeError):
            comp.num_sphs

        comp.compute(box, num_neighbors, positions)

        # Test access
        comp.sph
        comp.num_particles
        comp.num_sphs

        self.assertEqual(comp.sph.shape[0], N*num_neighbors)

        self.assertEqual(comp.num_particles, positions.shape[0])

        self.assertEqual(comp.num_sphs/comp.num_particles, num_neighbors)

        self.assertEqual(comp.l_max, l_max)

    def test_global(self):
        N = 1000
        num_neighbors = 4
        l_max = 8
        L = 10

        box, positions = make_box_and_random_points(L, N)

        comp = freud.environment.LocalDescriptors(
            num_neighbors, l_max, .5, True)
        comp.compute(box, num_neighbors, positions, mode='global')

        sphs = comp.sph

        self.assertEqual(sphs.shape[0], N*num_neighbors)

    def test_particle_local(self):
        N = 1000
        num_neighbors = 4
        l_max = 8
        L = 10

        box, positions = make_box_and_random_points(L, N)
        orientations = np.random.uniform(-1, 1, size=(N, 4)).astype(np.float32)
        orientations /= np.sqrt(np.sum(orientations**2,
                                       axis=-1))[:, np.newaxis]

        comp = freud.environment.LocalDescriptors(
            num_neighbors, l_max, .5, True)

        with self.assertRaises(RuntimeError):
            comp.compute(box, num_neighbors, positions, mode='particle_local')

        comp.compute(box, num_neighbors, positions,
                     orientations=orientations, mode='particle_local')

        sphs = comp.sph

        self.assertEqual(sphs.shape[0], N*num_neighbors)

    def test_unknown_modes(self):
        N = 1000
        num_neighbors = 4
        l_max = 8
        L = 10

        box, positions = make_box_and_random_points(L, N)

        comp = freud.environment.LocalDescriptors(
            num_neighbors, l_max, .5, True)

        with self.assertRaises(RuntimeError):
            comp.compute(box, num_neighbors, positions,
                         mode='particle_local_wrong')

    def test_shape_twosets(self):
        N = 1000
        num_neighbors = 4
        l_max = 8
        L = 10

        box, positions = make_box_and_random_points(L, N)
        positions2 = np.random.uniform(-L/2, L/2,
                                       size=(N//3, 3)).astype(np.float32)

        comp = freud.environment.LocalDescriptors(
            num_neighbors, l_max, .5, True)
        comp.compute(box, num_neighbors, positions, positions2)
        sphs = comp.sph
        self.assertEqual(sphs.shape[0], N//3*num_neighbors)

    def test_repr(self):
        comp = freud.environment.LocalDescriptors(4, 8, 0.5, True)
        self.assertEqual(str(comp), str(eval(repr(comp))))

    def test_ql(self):
        """Check if we can reproduce Steinhardt Ql."""
        def get_Ql(p, descriptors, nlist):
            """Given a set of points and a LocalDescriptors object (and the
            underlying neighborlist), compute the per-particle Steinhardt Ql
            order parameter for all :math:`l` values up to the maximum quantum
            number used in the computation of the descriptors."""
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
                steinhardt = freud.order.Steinhardt(L)
                steinhardt.compute(box, points, nlist=nl)
                npt.assert_array_almost_equal(steinhardt.order, Ql[:, L])

    def test_ql_weighted(self):
        """Check if we can reproduce Steinhardt Ql with bond weights."""
        np.random.seed(0)

        def get_Ql(p, descriptors, nlist):
            """Given a set of points and a LocalDescriptors object (and the
            underlying neighborlist), compute the per-particle Steinhardt Ql
            order parameter for all :math:`l` values up to the maximum quantum
            number used in the computation of the descriptors."""
            Qbar_lm = np.zeros((p.shape[0], descriptors.sph.shape[1]),
                               dtype=np.complex128)
            for i in range(p.shape[0]):
                indices = nlist.index_i == i
                Ylms = descriptors.sph[indices, :]
                weights = nlist.weights[indices, np.newaxis]
                weights /= np.sum(weights)
                Qbar_lm[i, :] = np.sum(Ylms * weights, axis=0)

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

            # Generate random weights for each bond
            nl.weights[:] = np.random.rand(len(nl.weights))

            Ql = get_Ql(points, ld, nl)

            # Test all allowable values of l.
            for L in range(2, l_max+1):
                steinhardt = freud.order.Steinhardt(L, weighted=True)
                steinhardt.compute(box, points, nlist=nl)
                npt.assert_array_almost_equal(steinhardt.order, Ql[:, L])

    @unittest.skipIf(sys.version_info < (3, 2),
                     "functools.lru_cache only supported on Python 3.2+")
    @skipIfMissing('sympy.physics.wigner')
    def test_wl(self):
        """Check if we can reproduce Steinhardt Wl."""
        from functools import lru_cache
        from sympy.physics.wigner import wigner_3j

        def lm_index(l, m):
            return l**2 + (m if m >= 0 else l - m)

        @lru_cache(maxsize=None)
        def get_wigner3j(l, m1, m2, m3):
            return float(wigner_3j(l, l, l, m1, m2, m3))

        def get_Wl(p, descriptors, nlist):
            """Given a set of points and a LocalDescriptors object (and the
            underlying neighborlist), compute the per-particle Steinhardt Wl
            order parameter for all :math:`l` values up to the maximum quantum
            number used in the computation of the descriptors."""
            Qbar_lm = np.zeros((p.shape[0], descriptors.sph.shape[1]),
                               dtype=np.complex128)

            num_neighbors = descriptors.sph.shape[0]/p.shape[0]
            for i in range(p.shape[0]):
                indices = nlist.index_i == i
                Qbar_lm[i, :] = np.sum(descriptors.sph[indices, :],
                                       axis=0)/num_neighbors

            Wl = np.zeros((Qbar_lm.shape[0], descriptors.l_max+1),
                          dtype=np.complex128)
            for i in range(Wl.shape[0]):
                for l in range(Wl.shape[1]):
                    for m1 in range(-l, l+1):
                        for m2 in range(max(-l-m1, -l), min(l-m1, l)+1):
                            m3 = -m1 - m2
                            # Manually add Condon-Shortley phase
                            phase = 1
                            for m in m1, m2, m3:
                                if m > 0 and m % 2 == 1:
                                    phase *= -1
                            Wl[i, l] += phase * get_wigner3j(l, m1, m2, m3) * \
                                Qbar_lm[i, lm_index(l, m1)] * \
                                Qbar_lm[i, lm_index(l, m2)] * \
                                Qbar_lm[i, lm_index(l, m3)]
            return Wl

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

            Wl = get_Wl(points, ld, nl)

            # Test all allowable values of l.
            for L in range(2, l_max+1):
                steinhardt = freud.order.Steinhardt(L, Wl=True)
                steinhardt.compute(box, points, nlist=nl)
                npt.assert_array_almost_equal(steinhardt.order, Wl[:, L])

    @skipIfMissing('scipy.special')
    def test_ld(self):
        """Verify the behavior of LocalDescriptors by explicitly calculating
        spherical harmonics manually and verifying them."""
        from scipy.special import sph_harm
        atol = 1e-5
        L = 8
        N = 100
        box, points = make_box_and_random_points(L, N)

        num_neighbors = 1
        r_max = 2
        l_max = 2

        # We want to provide the NeighborList ourselves since we need to use it
        # again later anyway.
        nn = freud.locality.NearestNeighbors(r_max, num_neighbors)
        nl = nn.compute(box, points).nlist

        ld = freud.environment.LocalDescriptors(
            num_neighbors, l_max, r_max)
        ld.compute(box, num_neighbors, points, mode='global', nlist=nl)

        # Loop over the sphs and compute them explicitly.
        for idx, (i, j) in enumerate(nl):
            bond = box.wrap(points[j] - points[i])
            r = np.linalg.norm(bond)
            theta = np.arccos(bond[2]/r)
            phi = np.arctan2(bond[1], bond[0])

            count = 0
            for l in range(l_max+1):
                for m in range(l+1):
                    # Explicitly calculate the spherical harmonic with scipy
                    # and check the output.  Arg order is theta, phi for scipy,
                    # but we need to pass the swapped angles because it uses
                    # the opposite convention from fsph (which LocalDescriptors
                    # uses internally).
                    scipy_val = sph_harm(m, l, phi, theta)
                    ld_val = (-1)**abs(m) * ld.sph[idx, count]
                    self.assertTrue(np.isclose(
                        scipy_val, ld_val, atol=atol),
                        msg=("Failed for l={}, m={}, x={}, y = {}"
                             "\ntheta={}, phi={}").format(
                            l, m, scipy_val, ld_val,
                            theta, phi))
                    count += 1

                for neg_m in range(1, l+1):
                    m = -neg_m
                    scipy_val = sph_harm(m, l, phi, theta)
                    ld_val = ld.sph[idx, count]
                    self.assertTrue(np.isclose(
                        scipy_val, ld_val, atol=atol),
                        msg=("Failed for l={}, m={}, x={}, y = {}"
                             "\ntheta={}, phi={}").format(
                            l, m, scipy_val, ld_val,
                            theta, phi))
                    count += 1

    @skipIfMissing('scipy.special')
    def test_ref_point_ne_points(self):
        """Verify the behavior of LocalDescriptors by explicitly calculating
        spherical harmonics manually and verifying them."""
        from scipy.special import sph_harm
        atol = 1e-5
        L = 8
        N = 100
        box, points = make_box_and_random_points(L, N)
        ref_points = np.random.rand(N, 3)*L - L/2

        num_neighbors = 1
        r_max = 2
        l_max = 2

        # We want to provide the NeighborList ourselves since we need to use it
        # again later anyway.
        nn = freud.locality.NearestNeighbors(r_max, num_neighbors)
        nl = nn.compute(box, ref_points, points).nlist

        ld = freud.environment.LocalDescriptors(
            num_neighbors, l_max, r_max)
        ld.compute(box, num_neighbors, ref_points, points, mode='global',
                   nlist=nl)

        # Loop over the sphs and compute them explicitly.
        for idx, (i, j) in enumerate(nl):
            bond = box.wrap(points[j] - ref_points[i])
            r = np.linalg.norm(bond)
            theta = np.arccos(bond[2]/r)
            phi = np.arctan2(bond[1], bond[0])

            count = 0
            for l in range(l_max+1):
                for m in range(l+1):
                    # Explicitly calculate the spherical harmonic with scipy
                    # and check the output.  Arg order is theta, phi for scipy,
                    # but we need to pass the swapped angles because it uses
                    # the opposite convention from fsph (which LocalDescriptors
                    # uses internally).
                    scipy_val = sph_harm(m, l, phi, theta)
                    ld_val = (-1)**abs(m) * ld.sph[idx, count]
                    self.assertTrue(np.isclose(
                        scipy_val, ld_val, atol=atol),
                        msg=("Failed for l={}, m={}, x={}, y = {}"
                             "\ntheta={}, phi={}").format(
                            l, m, scipy_val, ld_val,
                            theta, phi))
                    count += 1

                for neg_m in range(1, l+1):
                    m = -neg_m
                    scipy_val = sph_harm(m, l, phi, theta)
                    ld_val = ld.sph[idx, count]
                    self.assertTrue(np.isclose(
                        scipy_val, ld_val, atol=atol),
                        msg=("Failed for l={}, m={}, x={}, y = {}"
                             "\ntheta={}, phi={}").format(
                            l, m, scipy_val, ld_val,
                            theta, phi))
                    count += 1


if __name__ == '__main__':
    unittest.main()
