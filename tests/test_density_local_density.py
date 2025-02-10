# Copyright (c) 2010-2025 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np
import numpy.testing as npt
import pytest

import freud


def get_fraction(dist, r_max, diameter):
    """Compute what fraction of a point of the provided diameter at distance
    dist is contained in a sphere of radius r_max."""
    if dist < r_max - diameter / 2:
        return 1
    if dist > r_max + diameter / 2:
        return 0
    return -dist / diameter + r_max / diameter + 0.5


class TestLocalDensity:
    """Test fixture for LocalDensity"""

    def setup_method(self):
        """Initialize a box with randomly placed particles"""
        box_size = 10
        num_points = 10000
        self.box, self.pos = freud.data.make_random_system(
            box_size, num_points, seed=123
        )
        self.r_max = 3
        self.diameter = 1
        self.ld = freud.density.LocalDensity(self.r_max, self.diameter)

    def test_attribute_access(self):
        # Test attribute access before calling compute
        with pytest.raises(AttributeError):
            self.ld.density
        with pytest.raises(AttributeError):
            self.ld.num_neighbors
        with pytest.raises(AttributeError):
            self.ld.box

    def test_density(self):
        """Test that LocalDensity computes the correct density at each point"""

        r_max = self.r_max + 0.5 * self.diameter

        # This test is slow, so we only test AABBQuery.
        nq = freud.locality.AABBQuery(self.box, self.pos)
        neighbors = {"mode": "ball", "r_max": r_max, "exclude_ii": True}

        self.ld.compute(nq, neighbors=neighbors)

        # Test attribute access after calling compute
        self.ld.density
        self.ld.num_neighbors
        self.ld.box

        assert self.ld.box == freud.box.Box.cube(10)

        npt.assert_array_less(np.fabs(self.ld.density - 10.0), 1.5)

        npt.assert_array_less(np.fabs(self.ld.num_neighbors - 1130.973355292), 200)

    def test_ref_points(self):
        """Test that LocalDensity can compute a correct density at each point
        using the reference points as the data points."""
        self.ld.compute((self.box, self.pos))
        density = self.ld.density

        npt.assert_array_less(np.fabs(density - 10.0), 1.5)

        neighbors = self.ld.num_neighbors
        npt.assert_array_less(np.fabs(neighbors - 1130.973355292), 200)

    def test_repr(self):
        assert str(self.ld) == str(eval(repr(self.ld)))

    def test_points_ne_query_points(self):
        box = freud.box.Box.cube(10)
        points = np.array([[0, 0, 0], [1, 0, 0]])
        query_points = np.array([[0, 1, 0], [-1, -1, 0], [4, 0, 0]])
        diameter = 1
        r_max = 2

        v_around = 4 / 3 * (r_max**3) * np.pi

        ld = freud.density.LocalDensity(r_max, diameter)
        ld.compute((box, points), query_points)

        cd0 = 2 / v_around
        cd1 = (
            1
            + get_fraction(np.linalg.norm(points[1] - query_points[1]), r_max, diameter)
        ) / v_around
        correct_density = [cd0, cd1, 0]
        npt.assert_allclose(ld.density, correct_density, rtol=1e-4)

    def test_invalid_radial_distances(self):
        """Ensure that invalid r_max and diameter arguments raise errors."""
        with pytest.raises(ValueError):
            freud.density.LocalDensity(-1, 10)
        with pytest.raises(ValueError):
            freud.density.LocalDensity(2, -1)
