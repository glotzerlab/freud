import numpy as np
import numpy.testing as npt

import freud
import freud.locality


# class TestSANN:
#    def test_SANN(self):
def compute_SANN_neighborList(system, r_max):
    box, points = system
    N = len(points)
    aq = freud.locality.AABBQuery(box, points)
    nlist = aq.query(
        points, dict(mode="ball", r_max=r_max, exclude_ii=True)
    ).toNeighborList(sort_by_distance=True)
    sorted_neighbours = np.asarray(nlist)
    sorted_dist = np.asarray(nlist.distances)
    sol_id = []
    mask = np.zeros(len(nlist.distances), dtype=bool)
    for i in range(0, N):
        m = 3
        i_dist = sorted_dist[nlist.query_point_indices == i]
        while m < nlist.neighbor_counts[i] and np.sum(i_dist[:m]) / (m - 2) > i_dist[m]:
            m += 1
        mask[nlist.find_first_index(i) : nlist.find_first_index(i) + m] = True
        sol_id.append(m)
    sol_neighbors = sorted_neighbours[mask]
    sol_dist = sorted_dist[mask]
    solution_nlist = freud.locality.NeighborList.from_arrays(
        np.max(sol_neighbors[:, 0]) + 1,
        np.max(sol_neighbors[:, 1]) + 1,
        sol_neighbors[:, 0],
        sol_neighbors[:, 1],
        sol_dist,
    )
    return solution_nlist


def test_SANN_fcc():
    N_reap = 3
    r_max = 5.5
    # generate FCC crystal
    uc = freud.data.UnitCell.fcc()
    box, points = uc.generate_system(N_reap, scale=5)
    known_sol = compute_SANN_neighborList((box, points), r_max)
    f_SANN = freud.locality.FilterSANN()
    f_SANN.compute((box, points), {"r_max": r_max})
    sol = f_SANN.filtered_nlist
    npt.assert_allclose(sol.distances, known_sol.distances)
    npt.assert_allclose(sol.point_indices, known_sol.point_indices)
    npt.assert_allclose(sol.query_point_indices, known_sol.query_point_indices)


def test_SANN_simple():
    r_max = 1.5
    # generate FCC crystal
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    box = freud.box.Box.cube(10)
    known_sol = compute_SANN_neighborList((box, points), r_max)
    f_SANN = freud.locality.FilterSANN()
    f_SANN.compute((box, points), {"r_max": r_max, "exclude_ii": True})
    sol = f_SANN.filtered_nlist
    npt.assert_allclose(sol.distances, known_sol.distances)
    npt.assert_allclose(sol.point_indices, known_sol.point_indices)
    npt.assert_allclose(sol.query_point_indices, known_sol.query_point_indices)
