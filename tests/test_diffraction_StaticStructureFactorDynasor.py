import matplotlib
import numpy as np
import numpy.testing as npt
import pytest

import freud

matplotlib.use("agg")


class TestStaticStructureFactorDirect:
    def test_compute(self):
        sf = freud.diffraction.StaticStructureFactorDirect(1000, 100, 80000)
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        sf.compute((box, positions))

    def test_debye_validation(self):
        """Validate the Direct method against Debye method implementation."""
        bins = 100
        k_max = 100
        max_k_points = 20000
        sf_direct = freud.diffraction.StaticStructureFactorDirect(
            bins=bins, k_max=k_max, max_k_points=max_k_points
        )
        sf_debye = freud.diffraction.StaticStructureFactorDebye(
            bins=bins, k_max=k_max, k_min=0
        )
        box, points = freud.data.UnitCell.fcc().generate_system(4, sigma_noise=0.01)
        system = freud.locality.NeighborQuery.from_system((box, points))
        sf_direct.compute(system)
        sf_debye.compute(system)
        npt.assert_allclose(sf_direct.bin_centers, sf_debye.bin_centers)
        npt.assert_allclose(sf_direct.bin_edges, sf_debye.bin_edges)
        npt.assert_allclose(sf_direct.S_k[0], sf_debye.S_k[0], rtol=1e-5, atol=1e-5)

    # TODO: enable if N_total is needed
    #    def test_partial_structure_factor_arguments(self):
    #        sf = freud.diffraction.StaticStructureFactorDirect(1000, 100, 80000)
    #        box, positions = freud.data.UnitCell.fcc().generate_system(4)
    #        # Require N_total if and only if query_points are provided
    #        with pytest.raises(ValueError):
    #            sf.compute((box, positions), query_points=positions)
    #        with pytest.raises(ValueError):
    #            sf.compute((box, positions))

    def test_partial_structure_factor_symmetry(self):
        """Compute a partial structure factor and ensure it is symmetric under
        type exchange."""
        L = 10
        N = 1000
        max_k_points = 80000
        box, points = freud.data.make_random_system(L, N)
        system = freud.AABBQuery.from_system((box, points))
        A_points = system.points[: N // 3]
        B_points = system.points[N // 3 :]
        sf = freud.diffraction.StaticStructureFactorDirect(
            bins=100, k_max=10, max_k_points=max_k_points
        )
        sf.compute((system.box, B_points), query_points=A_points)
        S_AB = sf.S_k
        sf.compute((system.box, A_points), query_points=B_points)
        S_BA = sf.S_k
        npt.assert_allclose(S_AB, S_BA)

    def test_partial_structure_factor_sum_normalization(self):
        """Ensure that the weighted sum of the partial structure factors is
        equal to the full scattering."""
        L = 10
        N = 1000
        max_k_points = 80000
        box, points = freud.data.make_random_system(L, N)
        system = freud.AABBQuery.from_system((box, points))
        N_A = N // 3
        N_B = N - N_A
        A_points = system.points[:N_A]
        B_points = system.points[N_A:]
        sf = freud.diffraction.StaticStructureFactorDirect(
            bins=100, k_max=10, max_k_points=max_k_points
        )
        S_total = sf.compute(system).S_k
        S_total_as_partial = sf.compute(system, query_points=system.points).S_k
        npt.assert_allclose(S_total, S_total_as_partial)
        S_AA = sf.compute((system.box, A_points), query_points=A_points).S_k
        S_AB = sf.compute((system.box, B_points), query_points=A_points).S_k
        S_BA = sf.compute((system.box, A_points), query_points=B_points).S_k
        S_BB = sf.compute((system.box, B_points), query_points=B_points).S_k
        S_partial_sum = (
            1
            + (N_A / N) ** 2 * (S_AA - 1)
            + (N_A / N) * (N_B / N) * (S_AB - 1)
            + (N_B / N) * (N_A / N) * (S_BA - 1)
            + (N_B / N) ** 2 * (S_BB - 1)
        )
        npt.assert_allclose(S_total, S_partial_sum, rtol=1e-5, atol=1e-5)

    def test_large_k_partial_cross_term_goes_to_zero(self):
        """Ensure S_{AB}(k) goes to zero at large k."""
        L = 10
        N = 1000
        max_k_points = 80000
        box, points = freud.data.make_random_system(L, N)
        system = freud.AABBQuery.from_system((box, points))
        A_points = system.points[: N // 3]
        B_points = system.points[N // 3 :]
        sf = freud.diffraction.StaticStructureFactorDirect(
            bins=5, k_max=1e6, max_k_points=max_k_points
        )
        S_AB = sf.compute((system.box, B_points), query_points=A_points).S_k
        npt.assert_allclose(S_AB, 0, rtol=1e-5, atol=1e-5)

    def test_large_k_partial_self_term_goes_to_fraction(self):
        """Ensure S_{AA}(k) goes to N_A / N_total at large k."""
        L = 10
        N = 1000
        max_k_points = 80000
        box, points = freud.data.make_random_system(L, N)
        system = freud.AABBQuery.from_system((box, points))
        N_A = N // 3
        A_points = system.points[:N_A]
        sf = freud.diffraction.StaticStructureFactorDirect(
            bins=5, k_max=1e6, max_k_points=max_k_points
        )
        S_AA = sf.compute((system.box, A_points), query_points=A_points).S_k
        npt.assert_allclose(S_AA, N_A / N, rtol=1e-5, atol=1e-5)

    def test_large_k_scattering_goes_to_one(self):
        """Ensure S(k) goes to one at large k."""
        L = 10
        N = 1000
        max_k_points = 80000
        box, points = freud.data.make_random_system(L, N)
        system = freud.AABBQuery.from_system((box, points))
        sf = freud.diffraction.StaticStructureFactorDirect(
            bins=5, k_max=1e6, max_k_points=max_k_points
        )
        sf.compute(system)
        npt.assert_allclose(sf.S_k, 1, rtol=1e-5, atol=1e-5)

    def test_attribute_access(self):
        bins = 100
        k_max = 123
        max_k_points = 80000
        sf = freud.diffraction.StaticStructureFactorDirect(bins, k_max, max_k_points)
        assert sf.nbins == bins
        assert np.isclose(sf.k_max, k_max)

        box, positions = freud.data.UnitCell.fcc().generate_system(4)

        with pytest.raises(AttributeError):
            sf.S_k
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

    def test_attribute_shapes(self):
        bins = 100
        k_max = 123
        max_k_points = 80000
        sf = freud.diffraction.StaticStructureFactorDirect(bins, k_max, max_k_points)
        assert sf.bin_centers.shape == (bins,)
        assert sf.bin_edges.shape == (bins + 1,)
        npt.assert_allclose(sf.bounds, (sf.k_min, k_max))
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        sf.compute((box, positions))
        assert sf.S_k.shape == (bins,)

    def test_repr(self):
        bins = 100
        k_max = 123
        max_k_points = 80000
        sf = freud.diffraction.StaticStructureFactorDirect(bins, k_max, max_k_points)
        assert str(sf) == str(eval(repr(sf)))
