import matplotlib
import numpy as np
import numpy.testing as npt
import pytest
from numpy.lib import NumpyVersion
from StructureFactor_helper import (
    helper_partial_structure_factor_arguments,
    helper_test_attribute_access,
    helper_test_attribute_shapes,
    helper_test_bin_precission,
    helper_test_compute,
    helper_test_k_min,
    helper_test_large_k_partial_cross_term_goes_to_fraction,
    helper_test_large_k_partial_cross_term_goes_to_one,
    helper_test_large_k_partial_cross_term_goes_to_zero,
    helper_test_min_valid_k,
    helper_test_partial_structure_factor_sum_normalization,
    helper_test_partial_structure_factor_symmetry,
    helper_test_repr,
)

import freud

matplotlib.use("agg")


class TestStaticStructureFactorDirect:
    def test_compute(self):
        sf = freud.diffraction.StaticStructureFactorDirect(
            bins=1000, k_max=100, k_min=0, num_sampled_k_points=80000
        )
        helper_test_compute(sf)

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

    def test_S_0_is_N(self):
        # The Direct method evaluates S(k) in bins. Here, we choose the binning
        # parameters such that the first bin contains only the origin in k-space
        # and no other k-points. Thus the smallest bin is measuring S(0) = N.
        sf = freud.diffraction.StaticStructureFactorDirect(bins=100, k_max=10)
        L = 10
        N = 1000
        box, points = freud.data.make_random_system(L, N)
        system = freud.AABBQuery.from_system((box, points))
        sf.compute(system)
        assert np.isclose(sf.S_k[0], N)

    def test_accumulation(self):
        # This test ensures that accumulation and resetting works as expected.
        # See notes on test_S_0_is_N.
        sf = freud.diffraction.StaticStructureFactorDirect(bins=100, k_max=10)
        L = 10
        N = 1000
        # Ensure that accumulation averages correctly over different numbers of
        # points. We test N points, N*2 points, and N*3 points. On average, the
        # number of points is N * 2.
        for i in range(1, 4):
            box, points = freud.data.make_random_system(L, N * i)
            sf.compute((box, points), reset=False)
        print(sf.S_k[0], N * 2)
        assert np.isclose(sf.S_k[0], N * 2)
        box, points = freud.data.make_random_system(L, N * 2)
        sf.compute((box, points), reset=True)
        print(sf.S_k[0], N * 2)
        assert np.isclose(sf.S_k[0], N * 2)

    def test_k_min(self):
        sf1 = freud.diffraction.StaticStructureFactorDirect(bins=100, k_max=10)
        sf2 = freud.diffraction.StaticStructureFactorDirect(bins=50, k_max=10, k_min=5)
        helper_test_k_min(sf1, sf2)
        with pytest.raises(ValueError):
            freud.diffraction.StaticStructureFactorDirect(bins=100, k_max=10, k_min=-1)

    def test_partial_structure_factor_arguments(self):
        sf = freud.diffraction.StaticStructureFactorDirect(bins=1000, k_max=100)
        helper_partial_structure_factor_arguments(sf)

    def test_partial_structure_factor_symmetry(self):
        """Compute a partial structure factor and ensure it is symmetric under
        type exchange."""
        sf = freud.diffraction.StaticStructureFactorDirect(bins=100, k_max=10, k_min=0)
        helper_test_partial_structure_factor_symmetry(sf)

    def test_partial_structure_factor_sum_normalization(self):
        """Ensure that the weighted sum of the partial structure factors is
        equal to the full scattering."""
        sf = freud.diffraction.StaticStructureFactorDirect(
            bins=100, k_max=10, num_sampled_k_points=80000
        )
        helper_test_partial_structure_factor_sum_normalization(sf)

    def test_large_k_partial_cross_term_goes_to_zero(self):
        """Ensure S_{AB}(k) goes to zero at large k."""
        sf = freud.diffraction.StaticStructureFactorDirect(
            bins=100, k_max=500, k_min=400, num_sampled_k_points=200000
        )
        helper_test_large_k_partial_cross_term_goes_to_zero(sf)

    def test_large_k_partial_self_term_goes_to_fraction(self):
        """Ensure S_{AA}(k) goes to N_A / N_total at large k."""
        sf = freud.diffraction.StaticStructureFactorDirect(
            bins=100, k_max=500, k_min=400, num_sampled_k_points=200000
        )
        helper_test_large_k_partial_cross_term_goes_to_fraction(sf)

    def test_large_k_scattering_goes_to_one(self):
        """Ensure S(k) goes to one at large k."""
        sf = freud.diffraction.StaticStructureFactorDirect(
            bins=100, k_max=500, k_min=400, num_sampled_k_points=200000
        )
        helper_test_large_k_partial_cross_term_goes_to_one(sf)

    def test_attribute_access(self):
        bins = 100
        k_max = 123
        k_min = 0.1
        num_sampled_k_points = 10000
        sf = freud.diffraction.StaticStructureFactorDirect(
            bins=bins,
            k_max=k_max,
            k_min=k_min,
            num_sampled_k_points=num_sampled_k_points,
        )
        assert np.isclose(sf.num_sampled_k_points, num_sampled_k_points)
        with pytest.raises(AttributeError):
            sf.k_points
        helper_test_attribute_access(sf, bins, k_max, k_min)

    @pytest.mark.skipif(
        NumpyVersion(np.__version__) < "1.15.0", reason="Requires numpy>=1.15.0."
    )
    def test_bin_precision(self):
        # Ensure bin edges and bounds are precise
        bins = 100
        k_max = 123
        k_min = 0.1
        num_sampled_k_points = 10000
        sf = freud.diffraction.StaticStructureFactorDirect(
            bins=bins,
            k_max=k_max,
            k_min=k_min,
            num_sampled_k_points=num_sampled_k_points,
        )
        helper_test_bin_precission(sf, bins, k_max, k_min)

    def test_min_valid_k(self):
        Lx = 10
        Ly = 8
        Lz = 7
        bins = 100
        k_max = 30
        k_min = 1
        num_sampled_k_points = 100000
        min_valid_k = 2 * np.pi / Lz
        sf = freud.diffraction.StaticStructureFactorDirect(
            bins=bins,
            k_min=k_min,
            k_max=k_max,
            num_sampled_k_points=num_sampled_k_points,
        )
        helper_test_min_valid_k(sf, min_valid_k, Lx, Ly, Lz)

    def test_attribute_shapes(self):
        bins = 100
        k_max = 123
        k_min = 0.1
        num_sampled_k_points = 10000
        sf = freud.diffraction.StaticStructureFactorDirect(
            bins, k_max, k_min, num_sampled_k_points
        )
        helper_test_attribute_shapes(sf, bins, k_max, k_min)

    def test_repr(self):
        sf = freud.diffraction.StaticStructureFactorDirect(100, 123, 0.1, 10000)
        helper_test_repr(sf)
