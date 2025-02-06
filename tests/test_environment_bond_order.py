# Copyright (c) 2010-2025 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np
import pytest
import rowan
import util
from test_managedarray import ManagedArrayTestBase

import freud


class TestBondOrder:
    def test_bond_order(self):
        """Test the bond order diagram calculation."""
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        quats = np.array([[1, 0, 0, 0]] * len(positions))

        r_max = 1.5
        num_neighbors = 12
        n_bins_theta = n_bins_phi = nbins = 6
        bo = freud.environment.BondOrder(nbins)

        # Test access
        with pytest.raises(AttributeError):
            bo.box
        with pytest.raises(AttributeError):
            bo.bond_order

        # Test that there are exactly 12 non-zero bins for a perfect FCC
        # structure.
        bo.compute(
            (box, positions),
            quats,
            neighbors={"num_neighbors": num_neighbors, "r_max": r_max},
        )
        op_value = bo.bond_order.copy()
        assert np.sum(op_value > 0) == 12

        # Test access
        bo.box
        bo.bond_order
        bo.bin_counts
        bo.bin_edges
        bo.bin_centers

        # Test all the basic attributes.
        assert bo.nbins[0] == n_bins_theta
        assert bo.nbins[1] == n_bins_phi
        assert bo.box == box
        assert np.allclose(
            bo.bin_centers[0], (2 * np.arange(n_bins_theta) + 1) * np.pi / n_bins_theta
        )
        assert np.allclose(
            bo.bin_centers[1],
            (2 * np.arange(n_bins_phi) + 1) * np.pi / (n_bins_phi * 2),
        )

        test_set = util.make_raw_query_nlist_test_set(
            box, positions, positions, "nearest", r_max, num_neighbors, True
        )
        for nq, neighbors in test_set:
            # Test that lbod gives identical results when orientations are the same.
            # TODO: Find a way to test a rotated system to ensure that lbod gives
            # the desired results.
            bo = freud.environment.BondOrder(nbins, mode="lbod")
            bo.compute(nq, quats, neighbors=neighbors, reset=False)
            assert np.allclose(bo.bond_order, op_value)

            # Test access
            bo.box
            bo.bond_order

            # Test that obcd gives identical results when orientations are the
            # same.
            bo = freud.environment.BondOrder(nbins, mode="obcd")
            bo.compute(nq, quats, neighbors=neighbors)
            assert np.allclose(bo.bond_order, op_value)

            # Test that normal bod looks ordered for randomized orientations.
            np.random.seed(10893)
            random_quats = rowan.random.rand(len(positions))
            bo = freud.environment.BondOrder(nbins)
            bo.compute(nq, random_quats, neighbors=neighbors)
            assert np.allclose(bo.bond_order, op_value)

            # Ensure that obcd looks random for the randomized orientations.
            bo = freud.environment.BondOrder(nbins, mode="obcd")
            bo.compute(nq, random_quats, neighbors=neighbors)
            assert not np.allclose(bo.bond_order, op_value)
            assert np.sum(bo.bond_order > 0) == bo.bond_order.size

            # Test that oocd shows exactly one peak when all orientations
            # are the same.
            bo = freud.environment.BondOrder(nbins, mode="oocd")
            bo.compute(nq, quats, neighbors=neighbors, reset=False)
            assert np.sum(bo.bond_order > 0) == 1
            assert bo.bond_order[0, 0] > 0

            # Test that oocd is highly disordered with random quaternions. In
            # practice, the edge bins may still not get any values, so just
            # check that we get a lot of values.
            bo = freud.environment.BondOrder(nbins, mode="oocd")
            bo.compute(nq, random_quats, neighbors=neighbors)
            assert np.sum(bo.bond_order > 0) > 30

    def test_repr(self):
        bo = freud.environment.BondOrder((6, 6))
        assert str(bo) == str(eval(repr(bo)))

    def test_points_ne_query_points(self):
        lattice_size = 10
        # big box to ignore periodicity
        box = freud.box.Box.square(lattice_size * 5)
        angle = np.pi / 30
        query_points, points = util.make_alternating_lattice(lattice_size, angle)

        r_max = 1.6

        num_neighbors = 12
        n_bins_theta = 35
        n_bins_phi = 2
        test_set = util.make_raw_query_nlist_test_set(
            box, points, query_points, "nearest", r_max, num_neighbors, False
        )
        for nq, neighbors in test_set:
            bod = freud.environment.BondOrder(bins=(n_bins_theta, n_bins_phi))

            # orientations are not used in bod mode
            orientations = np.array([[1, 0, 0, 0]] * len(points))
            query_orientations = np.array([[1, 0, 0, 0]] * len(query_points))

            bod.compute(
                nq, orientations, query_points, query_orientations, neighbors=neighbors
            )

            # we want to make sure that we get 12 nonzero places, so we can
            # test whether we are not considering neighbors between points
            assert np.count_nonzero(bod.bond_order) == 12
            assert len(np.unique(bod.bond_order)) == 2


class TestBondOrderManagedArray(ManagedArrayTestBase):
    def build_object(self):
        self.obj = freud.environment.BondOrder(bins=(10, 10))

    @property
    def computed_properties(self):
        return ["bond_order", "bin_counts"]

    def compute(self):
        box = freud.box.Box.square(10)
        num_points = 100
        points = np.random.rand(num_points, 3) * box.L - box.L / 2
        orientations = rowan.random.rand(num_points)
        self.obj.compute((box, points), orientations, neighbors={"r_max": 3})
