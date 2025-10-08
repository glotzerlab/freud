# Copyright (c) 2010-2025 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import os
import pathlib

import numpy as np
import numpy.testing as npt
import pytest
from util import sort_rounded_xyz_array

import freud


class TestUnitCell:
    def test_square(self):
        """Test that the square lattice is correctly generated."""
        box, points = freud.data.UnitCell.square().generate_system()
        assert box == freud.box.Box.square(1)
        npt.assert_array_equal(points, [[-0.5, -0.5, 0]])

    def test_sc(self):
        """Test that the sc lattice is correctly generated."""
        box, points = freud.data.UnitCell.sc().generate_system()
        assert box == freud.box.Box.cube(1)
        npt.assert_array_equal(points, [[-0.5, -0.5, -0.5]])

    def test_bcc(self):
        """Test that the bcc lattice is correctly generated."""
        box, points = freud.data.UnitCell.bcc().generate_system()
        assert box == freud.box.Box.cube(1)
        npt.assert_array_equal(points, [[0, 0, 0], [-0.5, -0.5, -0.5]])

    def test_fcc(self):
        """Test that the fcc lattice is correctly generated."""
        box, points = freud.data.UnitCell.fcc().generate_system()
        assert box == freud.box.Box.cube(1)
        npt.assert_array_equal(
            points, [[0, 0, -0.5], [0, -0.5, 0], [-0.5, 0, 0], [-0.5, -0.5, -0.5]]
        )

    @pytest.mark.parametrize(
        "fn",
        [pathlib.Path(os.path.realpath(__file__)).parent / "example_file.cif"],
    )
    def test_cif(self, fn):
        """Test that the data from cif files is correct"""
        EXPECTED_L = 3.6
        box, points = freud.data.UnitCell.from_cif(fn).generate_system()
        points /= EXPECTED_L

        # Boxes are equal within fp precision
        npt.assert_allclose(
            [*box.to_dict().values()],
            [*freud.box.Box.cube(EXPECTED_L).to_dict().values()],
            rtol=1e-15,
            atol=1e-15,
        )
        npt.assert_allclose(
            points[::-1],
            [[0, 0, -0.5], [0, -0.5, 0], [-0.5, 0, 0], [-0.5, -0.5, -0.5]],
            rtol=1e-15,
            atol=1e-15,
        )

    @pytest.mark.parametrize("scale", [0.5, 2])
    def test_scale(self, scale):
        """Test the generation of a scaled structure."""
        box, points = freud.data.UnitCell.fcc().generate_system(scale=scale)
        assert box == freud.box.Box.cube(scale)
        npt.assert_array_equal(
            points,
            scale
            * np.array([[0, 0, -0.5], [0, -0.5, 0], [-0.5, 0, 0], [-0.5, -0.5, -0.5]]),
        )

    @pytest.mark.parametrize("num_replicas", range(1, 10))
    def test_replicas(self, num_replicas):
        """Test that replication works."""
        box, points = freud.data.UnitCell.fcc().generate_system(
            num_replicas=num_replicas
        )
        assert box == freud.box.Box.cube(num_replicas)

        test_points = np.array(
            [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0], [0.0, 0.0, 0.0]]
        )
        test_points = test_points[np.newaxis, np.newaxis, np.newaxis, ...]
        test_points = np.tile(
            test_points, [num_replicas, num_replicas, num_replicas, 1, 1]
        )
        test_points[..., 0] += np.arange(num_replicas)[
            :, np.newaxis, np.newaxis, np.newaxis
        ]
        test_points[..., 1] += np.arange(num_replicas)[
            np.newaxis, :, np.newaxis, np.newaxis
        ]
        test_points[..., 2] += np.arange(num_replicas)[
            np.newaxis, np.newaxis, :, np.newaxis
        ]
        test_points = (test_points - (num_replicas * 0.5)).reshape(-1, 3)

        npt.assert_allclose(
            sort_rounded_xyz_array(points),
            sort_rounded_xyz_array(box.wrap(test_points)),
        )

    @pytest.mark.parametrize(
        "num_replicas",
        [0, 2.5, -1, [2, 2, 0], [2, 2, 2], "abc"],
    )
    def test_invalid_replicas(self, num_replicas):
        """Test that invalid replications raise errors."""
        with pytest.raises(ValueError):
            freud.data.UnitCell.square().generate_system(num_replicas=num_replicas)

    def test_noise(self):
        """Test that noise generation works."""
        sigma = 0.01
        box, points = freud.data.UnitCell.fcc().generate_system(
            sigma_noise=sigma, seed=0
        )
        assert box == freud.box.Box.cube(1)

        test_points = np.array(
            [[0, 0, -0.5], [0, -0.5, 0], [-0.5, 0, 0], [-0.5, -0.5, -0.5]]
        )

        deltas = np.linalg.norm(box.wrap(test_points - points), axis=-1)
        # Nothing should be exactly equal, but differences should not be too
        # large. 4 sigma is an arbitrary choice that gives a high probability
        # of the test passing (although not necessary since the seed is set
        # above).
        assert not np.allclose(deltas, 0)
        npt.assert_allclose(deltas, 0, atol=4 * sigma)

    def test_seed(self):
        """Ensure that seeding does not overwrite the global random state."""
        num_points = 10
        sigma = 0.01

        np.random.seed(0)
        first_rand = np.random.randint(num_points)
        _box, _points = freud.data.UnitCell.fcc().generate_system(
            sigma_noise=sigma, seed=1
        )
        second_rand = np.random.randint(num_points)

        np.random.seed(0)
        third_rand = np.random.randint(num_points)
        _box, _points = freud.data.UnitCell.fcc().generate_system(
            sigma_noise=sigma, seed=2
        )
        fourth_rand = np.random.randint(num_points)

        npt.assert_array_equal(first_rand, third_rand)
        npt.assert_array_equal(second_rand, fourth_rand)


class TestRandomSystem:
    @pytest.mark.parametrize("N", [0, 1, 10, 100, 1000])
    @pytest.mark.parametrize("is2D", [True, False])
    def test_sizes_and_dimensions(self, N, is2D):
        box, points = freud.data.make_random_system(
            box_size=10, num_points=N, is2D=is2D, seed=1
        )
        assert points.shape == (N, 3)
        assert box.is2D == is2D

    def test_seed(self):
        """Ensure that seeding does not overwrite the global random state."""
        box_size = 10
        num_points = 10

        np.random.seed(0)
        first_rand = np.random.randint(num_points)
        _box, _points = freud.data.make_random_system(
            box_size=box_size, num_points=num_points, seed=1
        )
        second_rand = np.random.randint(num_points)

        np.random.seed(0)
        third_rand = np.random.randint(num_points)
        _box, _points = freud.data.make_random_system(
            box_size=box_size, num_points=num_points, seed=2
        )
        fourth_rand = np.random.randint(num_points)

        npt.assert_array_equal(first_rand, third_rand)
        npt.assert_array_equal(second_rand, fourth_rand)
