from abc import abstractmethod

import numpy as np
import numpy.testing as npt

import freud
import freud.locality


class FilterTest:
    """Tests which are common to all filter classes."""

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
            filt.compute(sys, dict(r_max=1.5))
            assert filt.unfiltered_nlist is not None
            assert filt.filtered_nlist is not None


class TestRAD(FilterTest):
    def get_filters(self):
        return [freud.locality.FilterRAD()]

    @staticmethod
    def compute_RAD_neighborlist(system, r_max):

        box = system[0]
        points = system[1]

        N = len(points)
        aq = freud.locality.AABBQuery(*system)
        nlist = aq.query(
            points, dict(mode="ball", r_max=r_max, exclude_ii=True)
        ).toNeighborList(sort_by_distance=True)
        sorted_neighbors = np.asarray(nlist)
        list_of_neighs = []
        # loop over all particles
        for i in range(0, N):
            # put closest neighbors in the list of valid neighbors
            i_neighbors = sorted_neighbors[sorted_neighbors[:, 0] == i, 1]
            list_of_neighs.append([i, i_neighbors[0]])
            # loop over neighours starting from second
            for jj in range(1, len(i_neighbors)):
                j = i_neighbors[jj]
                is_a_good_neighbour = True
                # loop over all neighbors that are closer
                for k in i_neighbors[:jj]:
                    # check the condition
                    v1 = box.wrap(points[i] - points[j])
                    v2 = box.wrap(points[i] - points[k])
                    coz = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    # if true add to the list
                    if 1 / np.dot(v1, v1) < (coz / np.dot(v2, v2)):
                        is_a_good_neighbour = False
                        break
                if is_a_good_neighbour:
                    list_of_neighs.append([i, j])

        # make freud nlist from list of neighbors
        sorted_neighbors = np.array(list_of_neighs)
        dists = np.linalg.norm(
            box.wrap(points[sorted_neighbors[:, 0]] - points[sorted_neighbors[:, 1]]),
            axis=-1,
        )
        nlist = freud.locality.NeighborList.from_arrays(
            np.max(sorted_neighbors[:, 0] + 1),
            np.max(sorted_neighbors[:, 1]) + 1,
            sorted_neighbors[:, 0],
            sorted_neighbors[:, 1],
            dists,
        )

        return nlist

    def test_RAD_random(self):
        """Test different implementations vs a random system."""
        N = 100
        L = 10
        r_max = 4.9

        sys = freud.data.make_random_system(L, N)
        nlist_1 = self.compute_RAD_neighborlist(sys, r_max)
        filt = freud.locality.FilterRAD().compute(
            sys, dict(r_max=r_max, exclude_ii=True)
        )
        nlist_2 = filt.filtered_nlist

        npt.assert_allclose(nlist_1.distances, nlist_2.distances, rtol=1e-5)
        npt.assert_allclose(nlist_1.point_indices, nlist_2.point_indices)
        npt.assert_allclose(nlist_1.query_point_indices, nlist_2.query_point_indices)

    def test_RAD_BCC(self):
        uc = freud.data.UnitCell.bcc()
        N_reap = 3
        r_max = 1.49
        box, points = uc.generate_system(N_reap)
        aq = freud.locality.AABBQuery(box, points)
        startlist = aq.query(
            points, dict(mode="ball", r_max=r_max, exclude_ii=True)
        ).toNeighborList(sort_by_distance=True)
        rad = freud.locality.FilterRAD()
        rad.compute((box, points), neighbors=startlist)
        npt.assert_array_equal(
            rad.filtered_nlist.neighbor_counts,
            np.ones(len(rad.filtered_nlist.neighbor_counts)) * 14,
        )


class TestSANN(FilterTest):
    """Tests specific to the SANN filtering method."""

    def get_filters(self):
        return [freud.locality.FilterSANN()]

    @staticmethod
    def compute_SANN_neighborlist(system, r_max):
        """Compute SANN in python."""
        box, points = system
        N = len(points)
        aq = freud.locality.AABBQuery(box, points)
        nlist = aq.query(
            points, dict(mode="ball", r_max=r_max, exclude_ii=True)
        ).toNeighborList(sort_by_distance=True)
        sorted_neighbors = np.asarray(nlist)
        sorted_dist = np.asarray(nlist.distances)
        sol_id = []
        mask = np.zeros(len(nlist.distances), dtype=bool)
        for i in range(0, N):
            m = 3
            i_dist = sorted_dist[nlist.query_point_indices == i]
            while (
                m < nlist.neighbor_counts[i]
                and np.sum(i_dist[:m]) / (m - 2) > i_dist[m]
            ):
                m += 1
            mask[nlist.find_first_index(i) : nlist.find_first_index(i) + m] = True
            sol_id.append(m)
        sol_neighbors = sorted_neighbors[mask]
        sol_dist = sorted_dist[mask]
        solution_nlist = freud.locality.NeighborList.from_arrays(
            np.max(sol_neighbors[:, 0]) + 1,
            np.max(sol_neighbors[:, 1]) + 1,
            sol_neighbors[:, 0],
            sol_neighbors[:, 1],
            sol_dist,
        )
        return solution_nlist

    def test_SANN_random(self):
        """Test different implementations vs a random system."""
        N = 100
        L = 10
        r_max = 4.9

        sys = freud.data.make_random_system(L, N)
        nlist_1 = self.compute_SANN_neighborlist(sys, r_max)
        filtersann = freud.locality.FilterSANN().compute(
            sys, dict(r_max=r_max, exclude_ii=True)
        )
        nlist_2 = filtersann.filtered_nlist

        npt.assert_allclose(nlist_1.distances, nlist_2.distances)
        npt.assert_allclose(nlist_1.point_indices, nlist_2.point_indices)
        npt.assert_allclose(nlist_1.query_point_indices, nlist_2.query_point_indices)

    def test_SANN_fcc(self):
        """make sure python and cpp implementations agree for an FCC."""
        N_reap = 3
        r_max = 5.5
        # generate FCC crystal
        uc = freud.data.UnitCell.fcc()
        box, points = uc.generate_system(N_reap, scale=5)
        known_sol = self.compute_SANN_neighborlist((box, points), r_max)
        f_SANN = freud.locality.FilterSANN()
        f_SANN.compute((box, points), {"r_max": r_max})
        sol = f_SANN.filtered_nlist
        npt.assert_allclose(sol.distances, known_sol.distances)
        npt.assert_allclose(sol.point_indices, known_sol.point_indices)
        npt.assert_allclose(sol.query_point_indices, known_sol.query_point_indices)

    def test_SANN_simple(self):
        """Assert SANN is correct when we compute the neighbors by hand.

        In this case the neighbors are the same, but the filtered nlist is
        sorted by distance, while the unfiltered nlist is sorted by point index.
        """
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
        f_SANN = freud.locality.FilterSANN()
        f_SANN.compute((box, points), {"r_max": r_max, "exclude_ii": True})

        # check the filtered nlist is right
        sol = f_SANN.filtered_nlist
        distances = [
            1,
            1,
            1,
            1,
            np.sqrt(2),
            np.sqrt(2),
            1,
            1,
            1,
            np.sqrt(2),
            np.sqrt(2),
            1,
            np.sqrt(2),
            np.sqrt(2),
        ]
        point_indices = [1, 4, 5, 0, 4, 5, 4, 0, 2, 1, 5, 0, 1, 4]
        query_point_indices = [0, 0, 0, 1, 1, 1, 2, 4, 4, 4, 4, 5, 5, 5]
        npt.assert_allclose(sol.distances, distances)
        npt.assert_allclose(sol.point_indices, point_indices)
        npt.assert_allclose(sol.query_point_indices, query_point_indices)

        # check the unfiltered nlist is right
        sol = f_SANN.unfiltered_nlist
        distances = [
            1,
            1,
            1,
            1,
            np.sqrt(2),
            np.sqrt(2),
            1,
            1,
            np.sqrt(2),
            1,
            np.sqrt(2),
            1,
            np.sqrt(2),
            np.sqrt(2),
        ]
        point_indices = [1, 4, 5, 0, 4, 5, 4, 0, 1, 2, 5, 0, 1, 4]
        query_point_indices = [0, 0, 0, 1, 1, 1, 2, 4, 4, 4, 4, 5, 5, 5]
        npt.assert_allclose(sol.distances, distances)
        npt.assert_allclose(sol.point_indices, point_indices)
        npt.assert_allclose(sol.query_point_indices, query_point_indices)
