# Copyright (c) 2010-2026 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from abc import abstractmethod

import conftest
import numpy as np
import numpy.testing as npt
import pytest

import freud
import freud.locality


class FilterTest:
    """Tests which are common to all filter classes."""

    @abstractmethod
    def get_filter_object(
        self, allow_incomplete_shell=False, terminate_after_blocked=True
    ):
        """Get a filter object.

        Note:
            ``terminate_after_blocked`` is unused when called with a class`.FilterSANN`
            object.
        """
        pass

    def compute_python_neighborlist(
        self, box, points, r_max, terminate_after_blocked=True
    ):
        """Compute the neighborlist for the system in python.

        The base implementation here creates the unfiltered nlist sorted by distance.
        The derived classes should use this nlist as a starting point for their
        implementations.

        Note:
            ``terminate_after_blocked`` is unused when called with a class`.FilterSANN`
            object.
        """
        aq = freud.locality.AABBQuery(box, points)
        return aq.query(
            points, dict(mode="ball", r_max=r_max, exclude_ii=True)
        ).toNeighborList(sort_by_distance=True)

    @pytest.mark.parametrize("allow_incomplete_shell", [True, False])
    @pytest.mark.parametrize("terminate_after_blocked", [True, False])
    def test_compute_and_properties(
        self, allow_incomplete_shell, terminate_after_blocked
    ):
        """Call compute and access unfiltered and filtered nlist."""
        # define system
        L = 10
        N = 100
        sys = freud.data.make_random_system(L, N, seed=1)
        filter_ = self.get_filter_object(
            allow_incomplete_shell, terminate_after_blocked
        )
        filter_.compute(sys, dict(r_max=4.5))
        assert filter_.unfiltered_nlist is not None
        assert filter_.filtered_nlist is not None

    def test_incomplete_shell(self):
        """Make sure error is raised when neighbor shells are incomplete."""
        L = 10
        N = 5
        sys = freud.data.make_random_system(L, N, seed=1)
        filt = self.get_filter_object(allow_incomplete_shell=False)
        with pytest.raises(RuntimeError):
            filt.compute(sys, dict(r_max=1.2, exclude_ii=True))

    def test_no_query_args(self):
        """Test unfiltered nlist with default neighbors argument."""
        N = 100
        L = 10

        sys = freud.data.make_random_system(L, N, seed=1)
        filt = self.get_filter_object()
        filt.compute(sys)
        nlist = filt.unfiltered_nlist

        # unfiltered nlist should have all except ii pairs
        num_bonds = N * (N - 1)
        assert len(nlist.distances) == num_bonds
        assert len(nlist.point_indices) == num_bonds
        assert len(nlist.query_point_indices) == num_bonds

    @pytest.mark.parametrize("terminate_after_blocked", [False, True])
    @pytest.mark.parametrize(
        ("crystal_cls", "num_neighbors"), [("bcc", 14), ("fcc", 12)]
    )
    def test_known_crystals(self, terminate_after_blocked, crystal_cls, num_neighbors):
        """Test against perfect crystals with known numbers of neighbors."""
        uc = getattr(freud.data.UnitCell, crystal_cls)()
        N_reap = 3
        r_max = 1.49
        sys = uc.generate_system(N_reap)
        filt = self.get_filter_object(terminate_after_blocked=terminate_after_blocked)
        filt.compute(sys, neighbors=dict(r_max=r_max, exclude_ii=True))
        num_neighbors_array = filt.filtered_nlist.neighbor_counts
        npt.assert_array_equal(
            num_neighbors_array,
            num_neighbors * np.ones_like(num_neighbors_array),
        )

    @pytest.mark.parametrize("terminate_after_blocked", [False, True])
    def test_random_system(self, terminate_after_blocked):
        """Compare freud and pure python implementations on a random system."""
        N = 100
        L = 10
        r_max = 4.9

        sys = freud.data.make_random_system(L, N, seed=1)
        nlist_1 = self.compute_python_neighborlist(*sys, r_max, terminate_after_blocked)
        filt = self.get_filter_object(terminate_after_blocked=terminate_after_blocked)
        filt.compute(sys, dict(r_max=r_max, exclude_ii=True))
        nlist_2 = filt.filtered_nlist

        npt.assert_allclose(nlist_1.distances, nlist_2.distances, rtol=5e-5)
        npt.assert_allclose(nlist_1.point_indices, nlist_2.point_indices)
        npt.assert_allclose(nlist_1.query_point_indices, nlist_2.query_point_indices)

    @pytest.mark.parametrize("nlist_property", ["filtered_nlist", "unfiltered_nlist"])
    def test_nlist_lifetime(self, nlist_property):
        def _get_nlist(sys):
            filt = self.get_filter_object()
            filt.compute(sys)
            return getattr(filt, nlist_property)

        conftest.nlist_lifetime_check(_get_nlist)


class TestRAD(FilterTest):
    """Tests specific to the RAD filtering method."""

    def get_filter_object(
        self, allow_incomplete_shell=False, terminate_after_blocked=True
    ):
        """Get a FilterRAD object."""
        return freud.locality.FilterRAD(allow_incomplete_shell, terminate_after_blocked)

    def compute_python_neighborlist(
        self, box, points, r_max, terminate_after_blocked=True
    ):
        """Compute the RAD neighborlist in python."""
        nlist = super().compute_python_neighborlist(
            box, points, r_max, terminate_after_blocked
        )
        sorted_neighbors = np.asarray(nlist)

        list_of_neighs = []
        # loop over all particles
        for i in range(0, len(points)):
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
                elif terminate_after_blocked:
                    break

        # make freud nlist from list of neighbors
        sorted_neighbors = np.array(list_of_neighs)
        vecs = box.wrap(points[sorted_neighbors[:, 0]] - points[sorted_neighbors[:, 1]])
        return freud.locality.NeighborList.from_arrays(
            np.max(sorted_neighbors[:, 0] + 1),
            np.max(sorted_neighbors[:, 1]) + 1,
            sorted_neighbors[:, 0],
            sorted_neighbors[:, 1],
            vecs,
        )

    def test_RAD_simple(self):
        """Assert RAD is correct when we compute the neighbors by hand."""
        r_max = 2.5
        neighbors = dict(r_max=r_max, exclude_ii=True)
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 4.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 2.1],
            ]
        )
        box = freud.box.Box.cube(10)
        sys = (box, points)

        # Verify with terminate_after_blocked=True (RAD closed)
        f_RAD = freud.locality.FilterRAD(
            allow_incomplete_shell=True, terminate_after_blocked=True
        )
        f_RAD.compute(sys, neighbors)
        sol = f_RAD.filtered_nlist
        distances = [
            1,
            1,
            1.9,
            1,
            1,
            1.9,
            2.1,
        ]
        point_indices = [3, 3, 4, 0, 1, 2, 0]
        query_point_indices = [0, 1, 2, 3, 3, 4, 4]
        npt.assert_allclose(sol.distances, distances)
        npt.assert_allclose(sol.point_indices, point_indices)
        npt.assert_allclose(sol.query_point_indices, query_point_indices)

        # Verify with terminate_after_blocked=False (RAD open)
        f_RAD = freud.locality.FilterRAD(
            allow_incomplete_shell=True, terminate_after_blocked=False
        )
        f_RAD.compute(sys, neighbors)
        sol = f_RAD.filtered_nlist
        distances = [
            1,
            2.1,
            1,
            1.9,
            1,
            1,
            1.9,
            2.1,
        ]
        point_indices = [3, 4, 3, 4, 0, 1, 2, 0]
        query_point_indices = [0, 0, 1, 2, 3, 3, 4, 4]
        npt.assert_allclose(sol.distances, distances)
        npt.assert_allclose(sol.point_indices, point_indices)
        npt.assert_allclose(sol.query_point_indices, query_point_indices)


class TestSANN(FilterTest):
    """Tests specific to the SANN filtering method."""

    def get_filter_object(
        self, allow_incomplete_shell=False, terminate_after_blocked=True
    ):
        """Get a FilterSANN object."""
        return freud.locality.FilterSANN(allow_incomplete_shell)

    def compute_python_neighborlist(
        self, box, points, r_max, terminate_after_blocked=True
    ):
        """Compute the SANN neighborlist in python.

        Note:
            The ``terminate_after_blocked`` argument is there for API consistency
            with other python filter implementations, it is not used by the SANN
            algorithm.
        """
        nlist = super().compute_python_neighborlist(
            box, points, r_max, terminate_after_blocked
        )
        sorted_neighbors = np.asarray(nlist)

        sorted_dist = np.asarray(nlist.distances)
        sorted_vecs = np.asarray(nlist.vectors)
        sol_id = []
        mask = np.zeros(len(nlist.distances), dtype=bool)
        for i in range(0, len(points)):
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
        sol_vecs = sorted_vecs[mask]
        return freud.locality.NeighborList.from_arrays(
            np.max(sol_neighbors[:, 0]) + 1,
            np.max(sol_neighbors[:, 1]) + 1,
            sol_neighbors[:, 0],
            sol_neighbors[:, 1],
            sol_vecs,
        )

    def test_SANN_simple(self):
        """Assert SANN is correct when we compute the neighbors by hand.

        In this case the neighbors are the same, but the filtered nlist is
        sorted by distance, while the unfiltered nlist is sorted by point index.
        """
        r_max = 1.5
        neighbors = dict(r_max=r_max, exclude_ii=True)
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
        sys = box, points

        f_SANN = freud.locality.FilterSANN(allow_incomplete_shell=True)
        f_SANN.compute(sys, neighbors)

        # check the filtered nlist
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

        # check the unfiltered nlist
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
