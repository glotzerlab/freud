import matplotlib
import numpy as np
import numpy.testing as npt
import pytest

import freud

matplotlib.use("agg")


class TestStaticStructureFactorDirect:
    def test_compute(self):
        bins = 1000
        k_max = 100
        k_min = 0
        max_k_points = 80000
        sf = freud.diffraction.StaticStructureFactorDirect(
            bins=bins, k_max=k_max, k_min=k_min, max_k_points=max_k_points
        )
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        sf.compute((box, positions))

    @pytest.mark.xfail(reason="The Debye method appears to be inaccurate.")
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
        # bdice: Currently this test fails. The first bin disagrees. I don't
        # expect for the Debye method to agree with this isotropic grid
        # sampling method for most k-values. However, I think the first bin
        # disagrees because the Debye method's first bin is not centered at
        # k=0. I am reconsidering the conversation Dom and I had earlier about
        # whether the k-space histogram should be centered at k=0. It seems
        # natural for the first bin S(0) to be equal to N. Additionally, it
        # makes some physical sense to have a difference between histograms in
        # real space (which have their first bin's left edge at 0) and
        # histograms in k-space (which might have the first bin's *center* at
        # 0).
        npt.assert_allclose(sf_direct.S_k[0], sf_debye.S_k[0], rtol=1e-5, atol=1e-5)

    def test_S_0_is_N(self):
        # The Direct method evaluates S(k) in bins. Here, we choose the binning
        # parameters such that the first bin contains only the origin in k-space
        # and no other k-points. Thus the smallest bin is measuring S(0) = N.
        L = 10
        N = 1000
        box, points = freud.data.make_random_system(L, N)
        system = freud.AABBQuery.from_system((box, points))
        sf = freud.diffraction.StaticStructureFactorDirect(bins=100, k_max=10)
        sf.compute(system)
        assert np.isclose(sf.S_k[0], N)

    def test_k_min(self):
        L = 10
        N = 1000
        box, points = freud.data.make_random_system(L, N)
        system = freud.AABBQuery.from_system((box, points))
        sf1 = freud.diffraction.StaticStructureFactorDirect(bins=100, k_max=10)
        sf1.compute(system)
        sf2 = freud.diffraction.StaticStructureFactorDirect(bins=50, k_max=10, k_min=5)
        sf2.compute(system)
        npt.assert_allclose(sf1.bin_centers[50:], sf2.bin_centers, rtol=1e-6, atol=1e-6)
        npt.assert_allclose(sf1.bin_edges[50:], sf2.bin_edges, rtol=1e-6, atol=1e-6)
        npt.assert_allclose(sf1.S_k[50:], sf2.S_k, rtol=1e-6, atol=1e-6)
        with pytest.raises(ValueError):
            freud.diffraction.StaticStructureFactorDirect(bins=100, k_max=10, k_min=-1)

    def test_partial_structure_factor_arguments(self):
        sf = freud.diffraction.StaticStructureFactorDirect(bins=1000, k_max=100)
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        # Require N_total if and only if query_points are provided
        with pytest.raises(ValueError):
            sf.compute((box, positions), query_points=positions)
        with pytest.raises(ValueError):
            sf.compute((box, positions), N_total=len(positions))

    def test_partial_structure_factor_symmetry(self):
        """Compute a partial structure factor and ensure it is symmetric under
        type exchange."""
        L = 10
        N = 1000
        box, points = freud.data.make_random_system(L, N)
        system = freud.AABBQuery.from_system((box, points))
        A_points = system.points[: N // 3]
        B_points = system.points[N // 3 :]
        sf = freud.diffraction.StaticStructureFactorDirect(bins=100, k_max=10)
        sf.compute((system.box, B_points), query_points=A_points, N_total=N)
        S_AB = sf.S_k
        sf.compute((system.box, A_points), query_points=B_points, N_total=N)
        S_BA = sf.S_k
        npt.assert_allclose(S_AB, S_BA, rtol=1e-6, atol=1e-6)

    def test_partial_structure_factor_sum_normalization(self):
        """Ensure that the weighted sum of the partial structure factors is
        equal to the full scattering."""
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
        S_total = sf.compute(system).S_k
        S_total_as_partial = sf.compute(system).S_k
        npt.assert_allclose(S_total, S_total_as_partial, rtol=1e-6, atol=1e-6)
        S_AA = sf.compute((system.box, A_points), query_points=A_points, N_total=N).S_k
        S_AB = sf.compute((system.box, B_points), query_points=A_points, N_total=N).S_k
        S_BA = sf.compute((system.box, A_points), query_points=B_points, N_total=N).S_k
        S_BB = sf.compute((system.box, B_points), query_points=B_points, N_total=N).S_k
        S_partial_sum = S_AA + S_AB + S_BA + S_BB
        npt.assert_allclose(S_total, S_partial_sum, rtol=1e-5, atol=1e-5)

    @pytest.mark.skip(
        reason="This test requires too much memory for allocating k-points."
    )
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
            bins=100, k_max=1e3, max_k_points=max_k_points
        )
        S_AB = sf.compute((system.box, B_points), query_points=A_points).S_k
        npt.assert_allclose(S_AB, 0, rtol=1e-5, atol=1e-5)

    @pytest.mark.skip(
        reason="This test requires too much memory for allocating k-points."
    )
    def test_large_k_partial_self_term_goes_to_fraction(self):
        """Ensure S_{AA}(k) goes to N_A / N_total at large k."""
        L = 10
        N = 1000
        max_k_points = 10000
        box, points = freud.data.make_random_system(L, N)
        system = freud.AABBQuery.from_system((box, points))
        N_A = N // 3
        A_points = system.points[:N_A]
        sf = freud.diffraction.StaticStructureFactorDirect(
            bins=100, k_max=1e3, max_k_points=max_k_points
        )
        S_AA = sf.compute((system.box, A_points), query_points=A_points).S_k
        npt.assert_allclose(S_AA, N_A / N, rtol=1e-5, atol=1e-5)

    @pytest.mark.skip(
        reason="This test requires too much memory for allocating k-points."
    )
    def test_large_k_scattering_goes_to_one(self):
        """Ensure S(k) goes to one at large k."""
        L = 10
        N = 1000
        max_k_points = 10000
        box, points = freud.data.make_random_system(L, N)
        system = freud.AABBQuery.from_system((box, points))
        sf = freud.diffraction.StaticStructureFactorDirect(
            bins=100, k_max=5e2, max_k_points=max_k_points
        )
        sf.compute(system)
        npt.assert_allclose(sf.S_k[-5:], 1, rtol=1e-5, atol=1e-5)

    def test_attribute_access(self):
        bins = 100
        k_max = 123
        k_min = 0.1
        max_k_points = 10000
        sf = freud.diffraction.StaticStructureFactorDirect(
            bins=bins, k_max=k_max, k_min=k_min, max_k_points=max_k_points
        )
        assert sf.nbins == bins
        assert np.isclose(sf.k_max, k_max)
        assert np.isclose(sf.k_min, k_min)
        assert np.isclose(sf.max_k_points, max_k_points)

        box, positions = freud.data.UnitCell.fcc().generate_system(4)

        with pytest.raises(AttributeError):
            sf.S_k
        with pytest.raises(AttributeError):
            sf.k_points
        with pytest.raises(AttributeError):
            sf.plot()

        sf.compute((box, positions))
        S_k = sf.S_k
        sf.k_points
        sf.plot()
        sf._repr_png_()

        # Make sure old data is not invalidated by new call to compute()
        box2, positions2 = freud.data.UnitCell.bcc().generate_system(3)
        sf.compute((box2, positions2))
        assert not np.array_equal(sf.S_k, S_k)

    def test_attribute_shapes(self):
        bins = 100
        k_max = 123
        k_min = 0.1
        max_k_points = 10000
        sf = freud.diffraction.StaticStructureFactorDirect(
            bins, k_max, k_min, max_k_points
        )
        assert sf.bin_centers.shape == (bins,)
        assert sf.bin_edges.shape == (bins + 1,)
        npt.assert_allclose(sf.bounds, (k_min, k_max))
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        sf.compute((box, positions))
        assert sf.S_k.shape == (bins,)
        assert sf.k_points.shape[1] == 3

    def test_repr(self):
        bins = 100
        k_max = 123
        k_min = 0.1
        max_k_points = 10000
        sf = freud.diffraction.StaticStructureFactorDirect(
            bins, k_max, k_min, max_k_points
        )
        assert str(sf) == str(eval(repr(sf)))
