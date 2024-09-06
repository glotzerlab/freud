# Copyright (c) 2010-2024 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np
import numpy.testing as npt
import pytest
from numpy.lib import NumpyVersion
from scipy.special import j0

import freud


def _sf_params():
    params_list = []
    params_list.append((182, 7, 0.1, 2e3))
    params_list.append((100, 13, 0.456, 1e4))
    params_list.append((50, 10, 0, 3e4))
    params_list.append((100, 9, 0, 8e5))
    return params_list


@pytest.fixture(scope="module", params=_sf_params())
def sf_params(request):
    """tuple: bins, k_max, k_min, num_sampled_k_points."""
    return request.param


def _sf_params_kmin_zero():
    """The subset of sf_params where the k_min value is zero."""
    params_list = []
    for params in _sf_params():
        if params[2] == 0:
            params_list.append(params)
    return params_list


@pytest.fixture(scope="module", params=_sf_params_kmin_zero())
def sf_params_kmin_zero(request):
    """tuple: bins, k_max, k_min=0, num_sampled_k_points."""
    return request.param


class StaticStructureFactorTest:
    @classmethod
    def build_structure_factor_object(
        cls, bins, k_max, k_min=0, num_sampled_k_points=None
    ):
        msg = (
            "The build_structure_factor_object method must be implemented for "
            "each inheriting class."
        )
        raise RuntimeError(msg)

    @classmethod
    def get_min_valid_k(cls, Lx, Ly, Lz=None):
        min_length = np.min([Lx, Ly]) if Lz is None else np.min([Lx, Ly, Lz])
        return 2 * np.pi / min_length

    def test_compute(self, sf_params):
        """Ensure calling compute does not crash."""
        sf = self.build_structure_factor_object(*sf_params)
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        sf.compute((box, positions))

    def test_k_min_nonnegative(self):
        with pytest.raises(ValueError):
            self.build_structure_factor_object(100, 7, -1)

    def test_partial_structure_factor_arguments(self, sf_params):
        sf = self.build_structure_factor_object(*sf_params)
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        with pytest.raises(ValueError):
            sf.compute((box, positions), query_points=positions)
        with pytest.raises(ValueError):
            sf.compute((box, positions), N_total=len(positions))

    def test_partial_structure_factor_symmetry(self, sf_params):
        """Compute a partial structure factor and ensure it is symmetric under
        type exchange."""
        L = 10
        N = 1000
        sf = self.build_structure_factor_object(*sf_params)
        box, points = freud.data.make_random_system(L, N, seed=123)
        system = freud.AABBQuery.from_system((box, points))
        A_points = system.points[: N // 3]
        B_points = system.points[N // 3 :]
        sf.compute((system.box, B_points), query_points=A_points, N_total=N)
        S_AB = sf.S_k
        sf.compute((system.box, A_points), query_points=B_points, N_total=N)
        S_BA = sf.S_k
        npt.assert_allclose(S_AB, S_BA, rtol=1e-5, atol=1e-5)

    def test_partial_structure_factor_sum_normalization(self, sf_params):
        """Ensure that the weighted sum of the partial structure factors is
        equal to the full scattering."""
        L = 10
        N = 1000
        sf = self.build_structure_factor_object(*sf_params)
        box, points = freud.data.make_random_system(L, N, seed=123)
        system = freud.AABBQuery.from_system((box, points))
        A_points = system.points[: N // 3]
        B_points = system.points[N // 3 :]
        S_total = sf.compute(system).S_k
        S_total_as_partial = sf.compute(
            system, query_points=system.points, N_total=N
        ).S_k
        npt.assert_allclose(S_total, S_total_as_partial, rtol=1e-5, atol=1e-5)
        S_AA = sf.compute((system.box, A_points), query_points=A_points, N_total=N).S_k
        S_AB = sf.compute((system.box, B_points), query_points=A_points, N_total=N).S_k
        S_BA = sf.compute((system.box, A_points), query_points=B_points, N_total=N).S_k
        S_BB = sf.compute((system.box, B_points), query_points=B_points, N_total=N).S_k
        S_partial_sum = S_AA + S_AB + S_BA + S_BB
        npt.assert_allclose(S_total, S_partial_sum, rtol=1e-5, atol=1e-5)

    def test_large_k_partial_cross_term_goes_to_zero(self, large_k_params):
        """Ensure S_{AB}(k) goes to 0 at large k."""
        L = 10
        N = 1000
        sf = self.build_structure_factor_object(*large_k_params)
        box, points = freud.data.make_random_system(L, N)
        system = freud.AABBQuery.from_system((box, points))
        A_points = system.points[: N // 3]
        B_points = system.points[N // 3 :]
        S_AB = sf.compute((system.box, B_points), query_points=A_points, N_total=N).S_k
        npt.assert_allclose(np.mean(S_AB), 0, atol=2e-2, rtol=1e-5)

    def test_large_k_partial_self_term_goes_to_fraction(self, large_k_params):
        """Ensure S_{AA}(k) goes to N_A / N_total at large k."""
        L = 10
        N = 1000
        sf = self.build_structure_factor_object(*large_k_params)
        box, points = freud.data.make_random_system(L, N)
        system = freud.AABBQuery.from_system((box, points))
        N_A = N // 3
        A_points = system.points[:N_A]
        S_AA = sf.compute((system.box, A_points), query_points=A_points, N_total=N).S_k
        npt.assert_allclose(np.mean(S_AA), N_A / N, rtol=1e-5, atol=2e-2)

    def test_large_k_scattering_goes_to_one(self, large_k_params):
        """Ensure S(k) goes to one at large k."""
        L = 10
        N = 1000
        sf = self.build_structure_factor_object(*large_k_params)
        box, points = freud.data.make_random_system(L, N)
        system = freud.AABBQuery.from_system((box, points))
        sf.compute(system)
        npt.assert_allclose(np.mean(sf.S_k), 1, rtol=1e-5, atol=2e-2)

    def test_attribute_access(self, sf_params):
        """Ensure parameters are initialized properly."""
        bins, k_max, k_min, num_sampled_k_points = sf_params
        sf = self.build_structure_factor_object(*sf_params)

        # only test common attribute in the super implementation
        assert np.isclose(sf.k_max, k_max)
        assert np.isclose(sf.k_min, k_min)
        npt.assert_allclose(sf.bounds, (k_min, k_max), rtol=1e-5, atol=1e-5)
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        with pytest.raises(AttributeError):
            sf.S_k
        with pytest.raises(AttributeError):
            sf.min_valid_k
        with pytest.raises(AttributeError):
            sf.plot()
        sf.compute((box, positions))
        S_k = sf.S_k
        sf.plot()
        sf._repr_png_()
        # Make sure old data is not invalidated by new call to compute()
        box2, positions2 = freud.data.UnitCell.bcc().generate_system(3)
        sf.compute((box2, positions2))
        assert not np.array_equal(sf.S_k, S_k)

    def test_min_valid_k(self, sf_params):
        Lx = 10
        Ly = 8
        Lz = 7
        sf = self.build_structure_factor_object(*sf_params)
        min_valid_k = self.get_min_valid_k(Lx, Ly, Lz)
        box, points = freud.data.UnitCell(
            [Lx / 10, Ly / 10, Lz / 10, 0, 0, 0],
            basis_positions=[[0, 0, 0], [0.3, 0.25, 0.35]],
        ).generate_system(10)
        sf.compute((box, points))
        assert np.isclose(sf.min_valid_k, min_valid_k)

    def test_attribute_shapes(self, sf_params):
        """Ensure attributes have the right shape."""
        bins, k_max, k_min, num_sampled_k_points = sf_params

        sf = self.build_structure_factor_object(*sf_params)

        # only test the common attributes in the super implementation
        npt.assert_allclose(sf.bounds, (k_min, k_max), rtol=1e-5, atol=1e-5)
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        sf.compute((box, positions))
        assert sf.S_k.shape == (bins,)

    def test_repr(self):
        """Ensure string representation is right. Not parametrized because of
        floating point error."""
        sf = self.build_structure_factor_object(100, 123, 0.1, 1e5)
        assert str(sf) == str(eval(repr(sf)))

    def test_S_0_is_N(self, sf_params_kmin_zero):
        L = 10
        N = 1000
        sf = self.build_structure_factor_object(*sf_params_kmin_zero)
        box, points = freud.data.make_random_system(L, N)
        system = freud.AABBQuery.from_system((box, points))
        sf.compute(system)
        assert np.isclose(sf.S_k[0], N)

    def test_accumulation(self, sf_params_kmin_zero):
        L = 10
        N = 100
        sf = self.build_structure_factor_object(*sf_params_kmin_zero)
        # Ensure that accumulation averages correctly over different numbers of
        # points. We test N points, N*2 points, and N*3 points. On average, the
        # number of points is N * 2.
        for i in range(1, 4):
            box, points = freud.data.make_random_system(L, N * i)
            sf.compute((box, points), reset=False)
        assert np.isclose(sf.S_k[0], N * 2)
        box, points = freud.data.make_random_system(L, N * 2)
        sf.compute((box, points), reset=True)
        assert np.isclose(sf.S_k[0], N * 2)


class TestStaticStructureFactorDebye(StaticStructureFactorTest):
    @pytest.fixture
    def large_k_params(self):
        """tuple: bins, k_max, k_min."""
        return 5, 1e6, 1e5

    @classmethod
    def get_min_valid_k(cls, Lx, Ly, Lz=None):
        min_length = np.min([Lx, Ly]) if Lz is None else np.min([Lx, Ly, Lz])
        return 4 * np.pi / min_length

    def test_k_min(self, sf_params):
        L = 10
        N = 100

        bins, k_max, _, _ = sf_params
        bins = bins + 1
        upper_bins = bins // 2 + 1
        k_min = k_max / 2

        sf1 = self.build_structure_factor_object(bins, k_max)
        sf2 = self.build_structure_factor_object(upper_bins, k_max, k_min=k_min)
        box, points = freud.data.make_random_system(L, N)
        system = freud.AABBQuery.from_system((box, points))
        sf1.compute(system)
        sf2.compute(system)
        npt.assert_allclose(
            sf1.k_values[bins // 2 :], sf2.k_values, rtol=1e-6, atol=1e-6
        )
        npt.assert_allclose(sf1.S_k[bins // 2 :], sf2.S_k, rtol=1e-6, atol=1e-6)

    def test_attribute_access(self, sf_params):
        """Ensure parameters are initialized properly."""
        super().test_attribute_access(sf_params)

        bins, k_max, k_min, num_sampled_k_points = sf_params
        sf = self.build_structure_factor_object(*sf_params)
        assert sf.num_k_values == bins

    def test_bin_precision(self, sf_params):
        """Ensure bin edges and bounds are precise."""
        bins, k_max, k_min, num_sampled_k_points = sf_params
        sf = self.build_structure_factor_object(*sf_params)
        expected_k_values = np.linspace(k_min, k_max, bins)
        npt.assert_allclose(sf.k_values, expected_k_values, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(
            sf.bounds,
            ([k_min, k_max]),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_attribute_shapes(self, sf_params):
        """Ensure attributes have the right shape."""
        super().test_attribute_shapes(sf_params)

        bins, k_max, k_min, num_sampled_k_points = sf_params
        sf = self.build_structure_factor_object(*sf_params)

        assert sf.k_values.shape == (bins,)

    @classmethod
    def build_structure_factor_object(
        cls, bins, k_max, k_min=0, num_sampled_k_points=None
    ):
        return freud.diffraction.StaticStructureFactorDebye(bins, k_max, k_min)

    @staticmethod
    def _validate_debye_method(system, bins, k_max, k_min):
        """Validation of the static structure calculation.

        This method is a pure Python reference implementation of the Debye method
        implemented in C++ in freud.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            bins (unsigned int):
                Number of bins in :math:`k` space.
            k_max (float):
                Maximum :math:`k` value to include in the calculation.
            k_min (float):
                Minimum :math:`k` value to include in the calculation.
        """
        system = freud.locality.NeighborQuery.from_system(system)
        N = len(system.points)

        Q = np.linspace(k_min, k_max, bins)
        S = np.zeros_like(Q)

        # Compute all pairwise distances
        distances = system.box.compute_all_distances(
            system.points, system.points
        ).flatten()

        for i, q in enumerate(Q):
            S[i] += np.sum(np.sinc(q * distances / np.pi)) / N

        return Q, S

    def test_debye_validation(self, sf_params):
        """Validate the Debye method against a Python implementation."""
        bins, k_max, k_min, _ = sf_params
        sf = self.build_structure_factor_object(*sf_params)
        box, points = freud.data.UnitCell.fcc().generate_system(4, sigma_noise=0.01)
        system = freud.locality.NeighborQuery.from_system((box, points))
        sf.compute(system)
        Q, S = self._validate_debye_method(system, bins, k_max, k_min)
        npt.assert_allclose(sf.k_values, Q, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(sf.S_k, S, rtol=1e-4, atol=1e-5)

    def test_debye_ase(self, sf_params_kmin_zero):
        """Validate Debye method agains ASE implementation."""
        ase = pytest.importorskip("ase")
        asexrd = pytest.importorskip("ase.utils.xrdebye")

        bins, k_max, k_min, _ = sf_params_kmin_zero
        box, points = freud.data.UnitCell.fcc().generate_system(4, sigma_noise=0.01)
        # ase implementation has no PBC taken into account
        box.periodic = False
        system = freud.locality.NeighborQuery.from_system((box, points))
        sf = freud.diffraction.StaticStructureFactorDebye(bins, k_max, k_min)
        sf.compute(system)
        # ASE system generation as atoms object
        atoms = ase.Atoms(
            positions=points, pbc=True, cell=box.L, numbers=np.ones(len(points))
        )
        xrd = asexrd.XrDebye(
            atoms=atoms, wavelength=1.0, method=None, damping=0.0, alpha=1.0
        )
        # calculate S_k for given set of k values
        S_ase = xrd.calc_pattern(sf.k_values, mode="SAXS") / len(points)
        npt.assert_allclose(sf.S_k, S_ase, rtol=1e-5, atol=1e-5)

    def test_2D(self):
        """Validate the Debye method in 2D."""

        # our own implementation of a 2D static structure factor, using scipy's
        # implementation of bessel j0
        def compute_ssf(k_points, positions, box):
            ssf = np.zeros_like(k_points)
            for pos1 in positions:
                for pos2 in positions:
                    dist = np.linalg.norm(box.wrap(pos1 - pos2))
                    ssf += j0(dist * k_points)
            ssf /= len(positions)
            return ssf

        L = 10
        N = 100
        sf = freud.diffraction.StaticStructureFactorDebye(num_k_values=100, k_max=10)
        box, points = freud.data.make_random_system(L, N, is2D=True)
        sf.compute((box, points))

        # compute structure factor using python implementation
        sf2 = compute_ssf(sf.k_values, points, box)

        # compare
        npt.assert_allclose(sf.S_k, sf2, rtol=1e-5, atol=1e-5)


class TestStaticStructureFactorDirect(StaticStructureFactorTest):
    @pytest.fixture
    def large_k_params(self):
        """tuple: bins, k_max, k_min, num_sampled_k_points."""
        return 100, 500, 400, 2e5

    def test_k_min(self, sf_params):
        L = 10
        N = 100

        bins, k_max, _, _ = sf_params
        upper_bins = bins // 2
        k_min = k_max / 2

        sf1 = self.build_structure_factor_object(bins, k_max)
        sf2 = self.build_structure_factor_object(upper_bins, k_max, k_min=k_min)
        box, points = freud.data.make_random_system(L, N)
        system = freud.AABBQuery.from_system((box, points))
        sf1.compute(system)
        sf2.compute(system)
        npt.assert_allclose(
            sf1.bin_centers[bins // 2 :], sf2.bin_centers, rtol=1e-6, atol=1e-6
        )
        npt.assert_allclose(
            sf1.bin_edges[bins // 2 :], sf2.bin_edges, rtol=1e-6, atol=1e-6
        )
        npt.assert_allclose(sf1.S_k[bins // 2 :], sf2.S_k, rtol=1e-6, atol=1e-6)

    def test_attribute_access(self, sf_params):
        """Ensure parameters are initialized properly."""
        super().test_attribute_access(sf_params)

        bins, k_max, k_min, num_sampled_k_points = sf_params
        sf = self.build_structure_factor_object(*sf_params)
        assert sf.num_sampled_k_points == num_sampled_k_points
        with pytest.raises(AttributeError):
            sf.k_points

    @pytest.mark.skipif(
        NumpyVersion(np.__version__) < "1.15.0", reason="Requires numpy>=1.15.0."
    )
    def test_bin_precision(self, sf_params):
        """Ensure bin edges and bounds are precise."""
        bins, k_max, k_min, num_sampled_k_points = sf_params
        sf = self.build_structure_factor_object(*sf_params)
        expected_bin_edges = np.histogram_bin_edges(
            np.array([0], dtype=np.float32), bins=bins, range=[k_min, k_max]
        )
        npt.assert_allclose(sf.bin_edges, expected_bin_edges, rtol=1e-5, atol=1e-5)
        expected_bin_centers = (expected_bin_edges[:-1] + expected_bin_edges[1:]) / 2
        npt.assert_allclose(sf.bin_centers, expected_bin_centers, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(
            sf.bounds,
            ([k_min, k_max]),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_attribute_shapes(self, sf_params):
        """Ensure attributes have the right shape."""
        super().test_attribute_shapes(sf_params)

        bins, k_max, k_min, num_sampled_k_points = sf_params
        sf = self.build_structure_factor_object(*sf_params)

        assert sf.bin_centers.shape == (bins,)
        assert sf.bin_edges.shape == (bins + 1,)

    @classmethod
    def build_structure_factor_object(
        cls, bins, k_max, k_min=0, num_sampled_k_points=0
    ):
        return freud.diffraction.StaticStructureFactorDirect(
            bins, k_max, k_min, num_sampled_k_points
        )

    def test_against_dynasor(self, sf_params_kmin_zero):
        """Validate the direct method agains dynasor package."""
        dsf_reciprocal = pytest.importorskip("dsf.reciprocal")
        binned_statistic = pytest.importorskip("scipy.stats").binned_statistic

        bins, k_max, k_min, num_sampled_k_points = sf_params_kmin_zero

        # Compute structure factor from freud
        sf_direct = freud.diffraction.StaticStructureFactorDirect(
            bins=bins, k_max=k_max, num_sampled_k_points=num_sampled_k_points
        )
        box, points = freud.data.UnitCell.fcc().generate_system(4, sigma_noise=0.01)
        system = freud.locality.NeighborQuery.from_system((box, points))
        sf_direct.compute(system)

        # Compute reference structure factor from dynasor package
        box_matrix = system.box.to_matrix()
        rec = dsf_reciprocal.reciprocal_isotropic(
            box_matrix, max_points=num_sampled_k_points, max_k=k_max
        )
        points_rho_ks = dsf_reciprocal.calc_rho_k(
            system.points.T, rec.k_points, ftype=rec.ftype
        )
        S_k_all = np.real(points_rho_ks * points_rho_ks.conjugate())
        S_k_binned, _, _ = binned_statistic(
            x=rec.k_distance,
            values=S_k_all,
            statistic="mean",
            bins=sf_direct.bin_edges,
        )
        npt.assert_allclose(sf_direct.S_k, S_k_binned, rtol=1e-5, atol=1e-5)
