import numpy as np
import numpy.testing as npt
from abc import abstractmethod

import freud
import freud.locality


class FilterTest:

    @abstractmethod
    def get_filters(self):
        """Return a list of filter objects to run tests on."""
        pass

    def test_compute_and_properties(self):
        """Call compute and access unfiltered and filtered nlist."""
        # define system
        L = 10
        N = 100
        sys = freud.data.make_random_system(L, N)
        filters = self.get_filters()
        for filt in filters:
            filt.compute(sys, dict(r_max=1.5, exclude_ii=True))
            assert filt.unfiltered_nlist is not None
            assert filt.filtered_nlist is not None


class TestSANN(FilterTest):

    def get_filters(self):
        return [freud.locality.FilterSANN()]

    @staticmethod
    def compute_SANN_neighborList(system, r_max):
        """Compute SANN in python."""
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

    def test_SANN_fcc(self):
        N_reap = 3
        r_max = 5.5
        # generate FCC crystal
        uc = freud.data.UnitCell.fcc()
        box, points = uc.generate_system(N_reap, scale=5)
        known_sol = self.compute_SANN_neighborList((box, points), r_max)
        f_SANN = freud.locality.FilterSANN()
        f_SANN.compute((box, points), {"r_max": r_max})
        sol = f_SANN.filtered_nlist
        npt.assert_allclose(sol.distances, known_sol.distances)
        npt.assert_allclose(sol.point_indices, known_sol.point_indices)
        npt.assert_allclose(sol.query_point_indices, known_sol.query_point_indices)

    def test_SANN_simple(self):
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
        known_sol = self.compute_SANN_neighborList((box, points), r_max)
        f_SANN = freud.locality.FilterSANN()
        f_SANN.compute((box, points), {"r_max": r_max, "exclude_ii": True})
        sol = f_SANN.filtered_nlist
        npt.assert_allclose(sol.distances, known_sol.distances)
        npt.assert_allclose(sol.point_indices, known_sol.point_indices)
        npt.assert_allclose(sol.query_point_indices, known_sol.query_point_indices)
