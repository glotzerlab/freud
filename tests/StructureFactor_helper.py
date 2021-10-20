import numpy as np
import numpy.testing as npt
import pytest

import freud


def helper_test_compute(sf):
    box, positions = freud.data.UnitCell.fcc().generate_system(4)
    sf.compute((box, positions))


def helper_test_S_0_is_N(sf):
    L = 10
    N = 1000
    box, points = freud.data.make_random_system(L, N)
    system = freud.AABBQuery.from_system((box, points))
    sf.compute(system)
    assert np.isclose(sf.S_k[0], N)


def helper_test_accumulation(sf):
    L = 10
    N = 1000
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


def helper_test_k_min(sf1, sf2):
    L = 10
    N = 1000
    box, points = freud.data.make_random_system(L, N)
    system = freud.AABBQuery.from_system((box, points))
    sf1.compute(system)
    sf2.compute(system)
    npt.assert_allclose(sf1.bin_centers[50:], sf2.bin_centers, rtol=1e-6, atol=1e-6)
    npt.assert_allclose(sf1.bin_edges[50:], sf2.bin_edges, rtol=1e-6, atol=1e-6)
    npt.assert_allclose(sf1.S_k[50:], sf2.S_k, rtol=1e-6, atol=1e-6)


def helper_partial_structure_factor_arguments(sf):
    box, positions = freud.data.UnitCell.fcc().generate_system(4)
    with pytest.raises(ValueError):
        sf.compute((box, positions), query_points=positions)
    with pytest.raises(ValueError):
        sf.compute((box, positions), N_total=len(positions))


def helper_test_partial_structure_factor_symmetry(sf):
    L = 10
    N = 1000
    box, points = freud.data.make_random_system(L, N, seed=123)
    system = freud.AABBQuery.from_system((box, points))
    A_points = system.points[: N // 3]
    B_points = system.points[N // 3 :]
    sf.compute((system.box, B_points), query_points=A_points, N_total=N)
    S_AB = sf.S_k
    sf.compute((system.box, A_points), query_points=B_points, N_total=N)
    S_BA = sf.S_k
    npt.assert_allclose(S_AB, S_BA, atol=1e-5, rtol=1e-5)


def helper_test_partial_structure_factor_sum_normalization(sf):
    L = 10
    N = 1000
    box, points = freud.data.make_random_system(L, N, seed=123)
    system = freud.AABBQuery.from_system((box, points))
    A_points = system.points[: N // 3]
    B_points = system.points[N // 3 :]
    S_total = sf.compute(system).S_k
    S_total_as_partial = sf.compute(system, query_points=system.points, N_total=N).S_k
    npt.assert_allclose(S_total, S_total_as_partial, rtol=1e-5, atol=1e-5)
    S_AA = sf.compute((system.box, A_points), query_points=A_points, N_total=N).S_k
    S_AB = sf.compute((system.box, B_points), query_points=A_points, N_total=N).S_k
    S_BA = sf.compute((system.box, A_points), query_points=B_points, N_total=N).S_k
    S_BB = sf.compute((system.box, B_points), query_points=B_points, N_total=N).S_k
    S_partial_sum = S_AA + S_AB + S_BA + S_BB
    npt.assert_allclose(S_total, S_partial_sum, rtol=1e-5, atol=1e-5)


def helper_test_large_k_partial_cross_term_goes_to_zero(sf):
    L = 10
    N = 1000
    box, points = freud.data.make_random_system(L, N)
    system = freud.AABBQuery.from_system((box, points))
    A_points = system.points[: N // 3]
    B_points = system.points[N // 3 :]
    S_AB = sf.compute((system.box, B_points), query_points=A_points, N_total=N).S_k
    npt.assert_allclose(np.mean(S_AB), 0, atol=2e-2, rtol=1e-5)


def helper_test_large_k_partial_cross_term_goes_to_fraction(sf):
    L = 10
    N = 1000
    box, points = freud.data.make_random_system(L, N)
    system = freud.AABBQuery.from_system((box, points))
    N_A = N // 3
    A_points = system.points[:N_A]
    S_AA = sf.compute((system.box, A_points), query_points=A_points, N_total=N).S_k
    npt.assert_allclose(np.mean(S_AA), N_A / N, rtol=1e-5, atol=2e-2)


def helper_test_large_k_partial_cross_term_goes_to_one(sf):
    L = 10
    N = 1000
    box, points = freud.data.make_random_system(L, N)
    system = freud.AABBQuery.from_system((box, points))
    sf.compute(system)
    npt.assert_allclose(np.mean(sf.S_k), 1, rtol=1e-5, atol=2e-2)


def helper_test_attribute_access(sf, bins, k_max, k_min):
    assert sf.nbins == bins
    assert np.isclose(sf.k_max, k_max)
    assert np.isclose(sf.k_min, k_min)
    npt.assert_allclose(sf.bounds, (k_min, k_max))
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


def helper_test_bin_precission(sf, bins, k_max, k_min):
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


def helper_test_min_valid_k(sf, min_valid_k, Lx, Ly, Lz):
    box, points = freud.data.UnitCell(
        [Lx / 10, Ly / 10, Lz / 10, 0, 0, 0],
        basis_positions=[[0, 0, 0], [0.3, 0.25, 0.35]],
    ).generate_system(10)
    sf.compute((box, points))
    assert np.isclose(sf.min_valid_k, min_valid_k)


def helper_test_attribute_shapes(sf, bins, k_max, k_min):
    assert sf.bin_centers.shape == (bins,)
    assert sf.bin_edges.shape == (bins + 1,)
    npt.assert_allclose(sf.bounds, (k_min, k_max))
    box, positions = freud.data.UnitCell.fcc().generate_system(4)
    sf.compute((box, positions))
    assert sf.S_k.shape == (bins,)


def helper_test_repr(sf):
    assert str(sf) == str(eval(repr(sf)))
