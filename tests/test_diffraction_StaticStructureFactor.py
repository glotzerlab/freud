import numpy as np
import numpy.testing as npt
import pytest
from numpy.lib import NumpyVersion

import freud


class StaticStructureFactorTest:
    @classmethod
    def build_structure_factor_object(
        cls, bins, k_max, k_min=0, num_sampled_k_points=None
    ):
        raise RuntimeError(
            "The build_structure_factor_object method must be implemented for "
            "each inheriting class."
        )

    def test_compute(self):
        sf = self.build_structure_factor_object(1000, 100, 0, 80000)
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        sf.compute((box, positions))

    def test_k_min(self):
        L = 10
        N = 1000
        sf1 = self.build_structure_factor_object(100, 10)
        sf2 = self.build_structure_factor_object(50, 10, k_min=5)
        box, points = freud.data.make_random_system(L, N)
        system = freud.AABBQuery.from_system((box, points))
        sf1.compute(system)
        sf2.compute(system)
        npt.assert_allclose(sf1.bin_centers[50:], sf2.bin_centers, rtol=1e-6, atol=1e-6)
        npt.assert_allclose(sf1.bin_edges[50:], sf2.bin_edges, rtol=1e-6, atol=1e-6)
        npt.assert_allclose(sf1.S_k[50:], sf2.S_k, rtol=1e-6, atol=1e-6)
        with pytest.raises(ValueError):
            self.build_structure_factor_object(100, 10, -1)

    def test_partial_structure_factor_arguments(self):
        sf = self.build_structure_factor_object(1000, 100)
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        with pytest.raises(ValueError):
            sf.compute((box, positions), query_points=positions)
        with pytest.raises(ValueError):
            sf.compute((box, positions), N_total=len(positions))

    def test_partial_structure_factor_symmetry(self):
        """Compute a partial structure factor and ensure it is symmetric under
        type exchange."""
        L = 10
        N = 1000
        sf = self.build_structure_factor_object(100, 10)
        box, points = freud.data.make_random_system(L, N, seed=123)
        system = freud.AABBQuery.from_system((box, points))
        A_points = system.points[: N // 3]
        B_points = system.points[N // 3 :]
        sf.compute((system.box, B_points), query_points=A_points, N_total=N)
        S_AB = sf.S_k
        sf.compute((system.box, A_points), query_points=B_points, N_total=N)
        S_BA = sf.S_k
        npt.assert_allclose(S_AB, S_BA, atol=1e-5, rtol=1e-5)

    def test_partial_structure_factor_sum_normalization(self):
        """Ensure that the weighted sum of the partial structure factors is
        equal to the full scattering."""
        L = 10
        N = 1000
        sf = self.build_structure_factor_object(100, 10, num_sampled_k_points=80000)
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

    def test_large_k_partial_cross_term_goes_to_zero(self):
        """Ensure S_{AB}(k) goes to 0 at large k."""
        L = 10
        N = 1000
        params = self.get_large_k_params()
        sf = self.build_structure_factor_object(**params)
        box, points = freud.data.make_random_system(L, N)
        system = freud.AABBQuery.from_system((box, points))
        A_points = system.points[: N // 3]
        B_points = system.points[N // 3 :]
        S_AB = sf.compute((system.box, B_points), query_points=A_points, N_total=N).S_k
        npt.assert_allclose(np.mean(S_AB), 0, atol=2e-2, rtol=1e-5)

    def test_large_k_partial_self_term_goes_to_fraction(self):
        """Ensure S_{AA}(k) goes to N_A / N_total at large k."""
        L = 10
        N = 1000
        params = self.get_large_k_params()
        sf = self.build_structure_factor_object(**params)
        box, points = freud.data.make_random_system(L, N)
        system = freud.AABBQuery.from_system((box, points))
        N_A = N // 3
        A_points = system.points[:N_A]
        S_AA = sf.compute((system.box, A_points), query_points=A_points, N_total=N).S_k
        npt.assert_allclose(np.mean(S_AA), N_A / N, rtol=1e-5, atol=2e-2)

    def test_large_k_scattering_goes_to_one(self):
        """Ensure S(k) goes to one at large k."""
        L = 10
        N = 1000
        params = self.get_large_k_params()
        sf = self.build_structure_factor_object(**params)
        box, points = freud.data.make_random_system(L, N)
        system = freud.AABBQuery.from_system((box, points))
        sf.compute(system)
        npt.assert_allclose(np.mean(sf.S_k), 1, rtol=1e-5, atol=2e-2)

    def test_attribute_access(self):
        bins = 100
        k_max = 123
        k_min = 0.1
        num_sampled_k_points = 10000
        sf = self.build_structure_factor_object(
            bins, k_max, k_min, num_sampled_k_points
        )
        assert sf.nbins == bins
        assert np.isclose(sf.k_max, k_max)
        assert np.isclose(sf.k_min, k_min)
        if hasattr(sf, "num_sampled_k_points"):
            assert sf.num_sampled_k_points == num_sampled_k_points
        npt.assert_allclose(sf.bounds, (k_min, k_max))
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        with pytest.raises(AttributeError):
            sf.S_k
        with pytest.raises(AttributeError):
            sf.min_valid_k
        with pytest.raises(AttributeError):
            sf.k_points
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

    @pytest.mark.skipif(
        NumpyVersion(np.__version__) < "1.15.0", reason="Requires numpy>=1.15.0."
    )
    def test_bin_precision(self):
        # Ensure bin edges and bounds are precise
        bins = 100
        k_max = 123
        k_min = 0.1
        num_sampled_k_points = 10000
        sf = self.build_structure_factor_object(
            bins, k_max, k_min, num_sampled_k_points
        )
        expected_bin_edges = np.histogram_bin_edges(
            np.array([0], dtype=np.float32), bins=bins, range=[k_min, k_max]
        )
        expected_bin_centers = (expected_bin_edges[:-1] + expected_bin_edges[1:]) / 2
        npt.assert_allclose(sf.bin_edges, expected_bin_edges, atol=1e-5, rtol=1e-5)
        npt.assert_allclose(sf.bin_centers, expected_bin_centers, atol=1e-5, rtol=1e-5)
        npt.assert_allclose(
            sf.bounds,
            ([expected_bin_edges[0], expected_bin_edges[-1]]),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_min_valid_k(self):
        Lx = 10
        Ly = 8
        Lz = 7
        bins = 100
        k_max = 30
        k_min = 1
        num_sampled_k_points = 100000
        sf = self.build_structure_factor_object(
            bins, k_max, k_min, num_sampled_k_points
        )
        min_valid_k = self.get_min_valid_k(Lx, Ly, Lz)
        box, points = freud.data.UnitCell(
            [Lx / 10, Ly / 10, Lz / 10, 0, 0, 0],
            basis_positions=[[0, 0, 0], [0.3, 0.25, 0.35]],
        ).generate_system(10)
        sf.compute((box, points))
        assert np.isclose(sf.min_valid_k, min_valid_k)

    def test_attribute_shapes(self):
        bins = 100
        k_max = 123
        k_min = 0.456
        num_sampled_k_points = 10000
        sf = self.build_structure_factor_object(
            bins, k_max, k_min, num_sampled_k_points
        )
        assert sf.bin_centers.shape == (bins,)
        assert sf.bin_edges.shape == (bins + 1,)
        npt.assert_allclose(sf.bounds, (k_min, k_max))
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        sf.compute((box, positions))
        assert sf.S_k.shape == (bins,)

    def test_repr(self):
        bins = 100
        k_max = 123
        k_min = 0.1
        num_sampled_k_points = 10000
        sf = self.build_structure_factor_object(
            bins, k_max, k_min, num_sampled_k_points
        )
        assert str(sf) == str(eval(repr(sf)))


class TestStaticStructureFactorDebye(StaticStructureFactorTest):
    @classmethod
    def build_structure_factor_object(
        cls, bins, k_max, k_min=0, num_sampled_k_points=None
    ):
        return freud.diffraction.StaticStructureFactorDebye(bins, k_max, k_min)

    @classmethod
    def get_large_k_params(cls):
        return {"bins": 5, "k_max": 1e6, "k_min": 1e5}

    @classmethod
    def get_min_valid_k(cls, Lx, Ly, Lz=None):
        if Lz == None:
            min_length = np.min([Lx, Ly])
        else:
            min_length = np.min([Lx, Ly, Lz])
        return 4 * np.pi / min_length

    def _validate_debye_method(self, system, bins, k_max, k_min):
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

        Q = np.linspace(k_min, k_max, bins, endpoint=False)
        Q += (k_max - k_min) / bins / 2
        S = np.zeros_like(Q)

        # Compute all pairwise distances
        distances = system.box.compute_all_distances(
            system.points, system.points
        ).flatten()

        for i, q in enumerate(Q):
            S[i] += np.sum(np.sinc(q * distances / np.pi)) / N

        return Q, S

    def test_debye_validation(self):
        """Validate the Debye method against a Python implementation."""
        bins = 1000
        k_max = 100
        k_min = 0
        sf = freud.diffraction.StaticStructureFactorDebye(bins, k_max, k_min)
        box, points = freud.data.UnitCell.fcc().generate_system(4, sigma_noise=0.01)
        system = freud.locality.NeighborQuery.from_system((box, points))
        sf.compute(system)
        Q, S = self._validate_debye_method(system, bins, k_max, k_min)
        npt.assert_allclose(sf.bin_centers, Q)
        npt.assert_allclose(sf.S_k, S, rtol=1e-5, atol=1e-5)

    def test_debye_ase(self):
        """Validate Debye method agains ASE implementation."""
        ase = pytest.importorskip("ase")
        asexrd = pytest.importorskip("ase.utils.xrdebye")
        bins = 1000
        k_max = 100
        k_min = 0
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
        S_ase = xrd.calc_pattern(sf.bin_centers, mode="SAXS") / len(points)
        npt.assert_allclose(sf.S_k, S_ase, rtol=1e-5, atol=1e-5)


class TestStaticStructureFactorDirect(StaticStructureFactorTest):
    @classmethod
    def build_structure_factor_object(
        cls, bins, k_max, k_min=0, num_sampled_k_points=0
    ):
        return freud.diffraction.StaticStructureFactorDirect(
            bins, k_max, k_min, num_sampled_k_points
        )

    @classmethod
    def get_large_k_params(cls):
        return {"bins": 100, "k_max": 500, "k_min": 400, "num_sampled_k_points": 200000}

    @classmethod
    def get_min_valid_k(cls, Lx, Ly, Lz=None):
        if Lz == None:
            min_length = np.min([Lx, Ly])
        else:
            min_length = np.min([Lx, Ly, Lz])
        return 2 * np.pi / min_length

    def test_S_0_is_N(self):
        L = 10
        N = 1000
        # The Direct method evaluates S(k) in bins. Here, we choose the binning
        # parameters such that the first bin contains only the origin in k-space
        # and no other k-points. Thus the smallest bin is measuring S(0) = N.
        sf = self.build_structure_factor_object(bins=100, k_max=10)
        box, points = freud.data.make_random_system(L, N)
        system = freud.AABBQuery.from_system((box, points))
        sf.compute(system)
        assert np.isclose(sf.S_k[0], N)

    def test_accumulation(self):
        L = 10
        N = 1000
        sf = self.build_structure_factor_object(bins=100, k_max=10)
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

    def test_against_dynasor(self):
        """Validate the direct method agains dynasor package."""
        dsf_reciprocal = pytest.importorskip("dsf.reciprocal")
        binned_statistic = pytest.importorskip("scipy.stats").binned_statistic
        bins = 100
        k_max = 30
        num_sampled_k_points = 20000

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
