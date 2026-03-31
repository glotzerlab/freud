# Copyright (c) 2010-2026 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import os
import pathlib

import numpy as np
import numpy.testing as npt
import pytest
from util import sort_rounded_xyz_array

import freud


class TestUnitCell:
    def test_rectangular(self):
        """Test that the rectangular lattice is correctly generated."""
        box, points = freud.data.UnitCell.rectangular().generate_system()
        assert box == freud.box.Box(1, 2)
        npt.assert_array_equal(points, [[-0.5, -1, 0]])

    @pytest.mark.parametrize("aspect", [0.5, 1.0, 2.0, 3.0])
    def test_rectangular_aspect(self, aspect):
        """Test that the rectangular lattice respects aspect ratio."""
        box, points = freud.data.UnitCell.rectangular(aspect=aspect).generate_system()
        assert box == freud.box.Box(1, aspect)
        npt.assert_array_equal(points, [[-0.5, -aspect / 2, 0]])

    def test_rectangular_centered(self):
        """Test that the centered rectangular lattice is correctly generated."""
        box, points = freud.data.UnitCell.rectangular(centered=True).generate_system()
        assert box == freud.box.Box(1, 2)
        npt.assert_array_equal(points, [[-0.5, -1, 0], [0, 0, 0]])

    @pytest.mark.parametrize("aspect", [1.0, 2.0, 4.0])
    def test_rectangular_centered_aspect(self, aspect):
        """Test centered rectangular with different aspect ratios."""
        box, points = freud.data.UnitCell.rectangular(
            aspect=aspect, centered=True
        ).generate_system()
        assert box == freud.box.Box(1, aspect)
        npt.assert_array_equal(points, [[-0.5, -aspect / 2, 0], [0, 0, 0]])

    def test_oblique(self):
        """Test that the oblique lattice is correctly generated."""
        box, points = freud.data.UnitCell.oblique().generate_system()
        # Default: aspect=1, theta=45 degrees
        theta_rad = np.deg2rad(45)
        expected_matrix = np.array(
            [[1, np.cos(theta_rad), 0], [0, np.sin(theta_rad), 0], [0, 0, 0]]
        )
        expected_box = freud.box.Box.from_matrix(expected_matrix)
        assert box == expected_box
        # Center of tilted box: (Lx/2 + xy*Ly/2, Ly/2, 0)
        center = np.array([box.Lx / 2 + box.xy * box.Ly / 2, box.Ly / 2, 0])
        npt.assert_allclose(points.squeeze(), -center, rtol=1e-6)

    @pytest.mark.parametrize("theta", [17.123, 30.0, 45.0, 60.0, 90.0])
    @pytest.mark.parametrize("aspect", [0.5, 1.0, 2.0])
    def test_oblique_params(self, aspect, theta):
        """Test that the oblique lattice respects aspect ratio and angle theta."""
        box, points = freud.data.UnitCell.oblique(
            aspect=aspect, theta=theta
        ).generate_system()
        theta_rad = np.deg2rad(theta)
        expected_matrix = np.array(
            [
                [1, aspect * np.cos(theta_rad), 0],
                [0, aspect * np.sin(theta_rad), 0],
                [0, 0, 0],
            ]
        )
        expected_box = freud.box.Box.from_matrix(expected_matrix)
        assert box == expected_box
        center = np.array([box.Lx / 2 + box.xy * box.Ly / 2, box.Ly / 2, 0])
        npt.assert_allclose(points.squeeze(), -center, rtol=1e-6)

    def test_graphene(self):
        """Test that the graphene lattice is correctly generated."""
        box, points = freud.data.UnitCell.graphene().generate_system()
        assert box == freud.box.Box(1, np.sqrt(3))
        expected_points = np.array(
            [
                [-0.5, -np.sqrt(3) / 2, 0],
                [-0.5, -np.sqrt(3) / 6, 0],
                [0, np.sqrt(3) / 3, 0],
                [0, 0, 0],
            ]
        )
        npt.assert_allclose(points, expected_points, rtol=1e-6)

    def test_graphene_bond_length(self):
        """Test that graphene has correct C-C bond length of 1/sqrt(3)."""
        # Generate a larger system so all neighbors are within the system
        box, points = freud.data.UnitCell.graphene().generate_system(3)
        bond_length = 1 / np.sqrt(3)
        # Use neighbor query to find 3 nearest neighbors for each atom
        aq = freud.locality.AABBQuery(box, points)
        nlist = aq.query(
            points, {"num_neighbors": 3, "exclude_ii": True}
        ).toNeighborList()
        # All bonds should have the same length (honeycomb structure)
        npt.assert_allclose(nlist.distances, bond_length, rtol=1e-5)

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

    def test_hcp(self):
        """Test that the HCP lattice is correctly generated."""
        uc = freud.data.UnitCell.hcp()
        box, points = uc.generate_system()
        c = np.sqrt(8 / 3)
        expected_box = freud.box.Box.from_box_lengths_and_angles(
            1, 1, c, np.pi / 2, np.pi / 2, np.deg2rad(120)
        )
        assert box == expected_box
        assert len(points) == 2

    def test_hcp_coordination(self):
        """Test that the HCP crystal has the expected coordination number of 12."""
        uc = freud.data.UnitCell.hcp()
        box, points = uc.generate_system(num_replicas=3)
        voro = freud.locality.Voronoi()
        voro.compute((box, points))
        nlist = voro.nlist
        # print(nlist.weights)
        npt.assert_array_equal(
            nlist.neighbor_counts,
            np.full(len(points), fill_value=12, dtype=int),
        )

    def test_hcp_voronoi_weights(self):
        """Test that the HCP crystal has the expected Voronoi facet area sqrt(2)/4.

        The Voronoi cell of HCP is the trapezo-rhombic dodecahedron, which is like if
        you cut a rhombic dodecahedron in half and twisted it by 60 degrees. This is
        similar to the construction of the pseudo-rhombicuboctahedron, but applied to a
        different shape.
        """
        uc = freud.data.UnitCell.hcp()
        box, points = uc.generate_system(num_replicas=3)
        voro = freud.locality.Voronoi()
        voro.compute((box, points))
        np.testing.assert_allclose(voro.nlist.weights, np.sqrt(2) / 4.0, atol=1e-6)

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
