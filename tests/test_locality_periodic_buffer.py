# Copyright (c) 2010-2026 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np
import numpy.testing as npt
import pytest

import freud


class TestPeriodicBuffer:
    def test_square(self):
        L = 10  # Box length
        N = 50  # Number of points

        box, positions = freud.data.make_random_system(L, N, is2D=True, seed=1)
        positions.flags["WRITEABLE"] = False

        pbuff = freud.locality.PeriodicBuffer()

        # Compute with zero buffer distance
        pbuff.compute((box, positions), buffer=0, images=False)
        assert len(pbuff.buffer_points) == 0
        assert len(pbuff.buffer_ids) == 0
        npt.assert_array_equal(pbuff.buffer_box.L, np.asarray(box.L))

        # Compute with buffer distances
        pbuff.compute((box, positions), buffer=0.5 * L, images=False)
        assert len(pbuff.buffer_points) == 3 * N
        assert len(pbuff.buffer_ids) == 3 * N
        npt.assert_array_equal(pbuff.buffer_box.L, 2 * np.asarray(box.L))

        # Compute with different buffer distances
        pbuff.compute((box, positions), buffer=[L, 0, 0], images=False)
        assert len(pbuff.buffer_points) == 2 * N
        assert len(pbuff.buffer_ids) == 2 * N
        npt.assert_array_equal(pbuff.buffer_box.L, box.L * np.array([3, 1, 1]))

        # Compute with zero images
        pbuff.compute((box, positions), buffer=0, images=True)
        assert len(pbuff.buffer_points) == 0
        assert len(pbuff.buffer_ids) == 0
        npt.assert_array_equal(pbuff.buffer_box.L, np.asarray(box.L))

        # Compute with images
        pbuff.compute((box, positions), buffer=1, images=True)
        assert len(pbuff.buffer_points) == 3 * N
        assert len(pbuff.buffer_ids) == 3 * N
        npt.assert_array_equal(pbuff.buffer_box.L, 2 * np.asarray(box.L))

        # Compute with different images
        pbuff.compute((box, positions), buffer=[1, 0, 0], images=True)
        assert len(pbuff.buffer_points) == N
        assert len(pbuff.buffer_ids) == N
        npt.assert_array_equal(pbuff.buffer_box.L, box.L * np.array([2, 1, 1]))

    def test_cube(self):
        L = 10  # Box length
        N = 50  # Number of points
        np.random.seed(0)

        box, positions = freud.data.make_random_system(L, N, is2D=False, seed=1)
        positions.flags["WRITEABLE"] = False

        pbuff = freud.locality.PeriodicBuffer()

        # Compute with zero buffer distance
        pbuff.compute((box, positions), buffer=0, images=False)
        assert len(pbuff.buffer_points) == 0
        assert len(pbuff.buffer_ids) == 0
        npt.assert_array_equal(pbuff.buffer_box.L, np.asarray(box.L))

        # Compute with buffer distances
        pbuff.compute((box, positions), buffer=0.5 * L, images=False)
        assert len(pbuff.buffer_points) == 7 * N
        assert len(pbuff.buffer_ids) == 7 * N
        npt.assert_array_equal(pbuff.buffer_box.L, 2 * np.asarray(box.L))

        # Compute with different buffer distances
        pbuff.compute((box, positions), buffer=[L, 0, L], images=False)
        assert len(pbuff.buffer_points) == 8 * N
        assert len(pbuff.buffer_ids) == 8 * N
        npt.assert_array_equal(pbuff.buffer_box.L, box.L * np.array([3, 1, 3]))

        # Compute with zero images
        pbuff.compute((box, positions), buffer=0, images=True)
        assert len(pbuff.buffer_points) == 0
        assert len(pbuff.buffer_ids) == 0
        npt.assert_array_equal(pbuff.buffer_box.L, np.asarray(box.L))

        # Compute with images
        pbuff.compute((box, positions), buffer=1, images=True)
        assert len(pbuff.buffer_points) == 7 * N
        assert len(pbuff.buffer_ids) == 7 * N
        npt.assert_array_equal(pbuff.buffer_box.L, 2 * np.asarray(box.L))

        # Compute with images-success
        pbuff.compute((box, positions), buffer=2, images=True)
        assert len(pbuff.buffer_points) == 26 * N
        assert len(pbuff.buffer_ids) == 26 * N
        npt.assert_array_equal(pbuff.buffer_box.L, 3 * np.asarray(box.L))

        # Compute with two images in x axis
        pbuff.compute((box, positions), buffer=np.array([1, 0, 0]), images=True)
        assert len(pbuff.buffer_points) == N
        assert len(pbuff.buffer_ids) == N
        npt.assert_array_equal(pbuff.buffer_box.Lx, 2 * np.asarray(box.Lx))

        # Compute with different images
        pbuff.compute((box, positions), buffer=[1, 0, 1], images=True)
        assert len(pbuff.buffer_points) == 3 * N
        assert len(pbuff.buffer_ids) == 3 * N
        npt.assert_array_equal(pbuff.buffer_box.L, box.L * np.array([2, 1, 2]))

    def test_fcc_unit_cell(self):
        s = np.sqrt(0.5)
        L = 2 * s  # Box length

        box = freud.box.Box.cube(L)  # Initialize box
        pbuff = freud.locality.PeriodicBuffer()
        positions = np.array([(s, s, 0), (s, 0, s), (0, s, s), (0, 0, 0)])
        positions.flags["WRITEABLE"] = False

        # Compute with zero buffer distance
        pbuff.compute((box, positions), buffer=0, images=False)
        assert len(pbuff.buffer_points) == 0
        assert len(pbuff.buffer_ids) == 0
        npt.assert_array_equal(pbuff.buffer_box.L, np.asarray(box.L))

        # Compute with buffer distances
        pbuff.compute((box, positions), buffer=0.5 * L, images=False)
        assert len(pbuff.buffer_points) == 7 * len(positions)
        assert len(pbuff.buffer_ids) == 7 * len(positions)
        npt.assert_array_equal(pbuff.buffer_box.L, 2 * np.asarray(box.L))

        """The test below looks like it should work the same as when using
        "images=True" with "buffer=L" but it fails due to numerical imprecision
        in the check determining whether a point is in the buffer box when
        there are points exactly on the boundary and an irrational box
        length such as sqrt(0.5), as in this test case.

        # Compute with buffer of one box length
        pbuff.compute((box, positions), buffer=L, images=False)
        self.assertEqual(len(pbuff.buffer_points), 8 * len(positions))
        npt.assert_array_equal(pbuff.buffer_box.L,
                               box.L * np.array([3, 1, 3]))
        """

        # Compute with zero images
        pbuff.compute((box, positions), buffer=0, images=True)
        assert len(pbuff.buffer_points) == 0
        assert len(pbuff.buffer_ids) == 0
        npt.assert_array_equal(pbuff.buffer_box.L, np.asarray(box.L))

        # Compute with images
        pbuff.compute((box, positions), buffer=1, images=True)
        assert len(pbuff.buffer_points) == 7 * len(positions)
        assert len(pbuff.buffer_ids) == 7 * len(positions)
        npt.assert_array_equal(pbuff.buffer_box.L, 2 * np.asarray(box.L))

        # Compute with images-success
        pbuff.compute((box, positions), buffer=2, images=True)
        assert len(pbuff.buffer_points) == 26 * len(positions)
        assert len(pbuff.buffer_ids) == 26 * len(positions)
        npt.assert_allclose(pbuff.buffer_box.L, 3 * np.asarray(box.L), atol=1e-6)

        # Compute with two images in x axis
        pbuff.compute((box, positions), buffer=np.array([1, 0, 0]), images=True)
        assert len(pbuff.buffer_points) == len(positions)
        assert len(pbuff.buffer_ids) == len(positions)
        npt.assert_array_equal(pbuff.buffer_box.Lx, 2 * np.asarray(box.Lx))

        # Compute with different images
        pbuff.compute((box, positions), buffer=[1, 0, 1], images=True)
        assert len(pbuff.buffer_points) == 3 * len(positions)
        assert len(pbuff.buffer_ids) == 3 * len(positions)
        npt.assert_array_equal(pbuff.buffer_box.L, box.L * np.array([2, 1, 2]))

    def test_triclinic(self):
        N = 50  # Number of points
        np.random.seed(0)

        box = freud.box.Box(Lx=2, Ly=2, Lz=2, xy=1, xz=0, yz=1)
        pbuff = freud.locality.PeriodicBuffer()

        # Generate random points in the box, in fractional coordinates
        positions = np.random.uniform(0, 1, size=(N, 3))

        # Convert fractional coordinates to real coordinates
        positions = np.asarray(list(map(box.make_absolute, positions)))
        positions = box.wrap(positions)

        # Compute with zero images
        pbuff.compute((box, positions), buffer=0, images=True)
        assert len(pbuff.buffer_points) == 0
        assert len(pbuff.buffer_ids) == 0
        npt.assert_array_equal(pbuff.buffer_box.L, np.asarray(box.L))

        # Compute with images
        pbuff.compute((box, positions), buffer=2, images=True)
        assert len(pbuff.buffer_points) == 26 * N
        assert len(pbuff.buffer_ids) == 26 * N
        npt.assert_array_equal(pbuff.buffer_box.L, 3 * np.asarray(box.L))

        # Compute with different images
        pbuff.compute((box, positions), buffer=[1, 0, 1], images=True)
        assert len(pbuff.buffer_points) == 3 * N
        assert len(pbuff.buffer_ids) == 3 * N
        npt.assert_array_equal(pbuff.buffer_box.L, box.L * np.array([2, 1, 2]))

    @pytest.mark.parametrize(("is2d", "points_fac"), [(True, 9), (False, 27)])
    def test_include_input_points(self, is2d, points_fac):
        L = 10  # Box length
        N = 50  # Number of points

        box, positions = freud.data.make_random_system(L, N, is2D=is2d, seed=1)
        positions.flags["WRITEABLE"] = False

        pbuff = freud.locality.PeriodicBuffer()
        pbuff.compute(
            (box, positions), buffer=2, images=True, include_input_points=True
        )

        assert len(pbuff.buffer_points) == points_fac * N

    def test_repr(self):
        pbuff = freud.locality.PeriodicBuffer()
        assert str(pbuff) == str(eval(repr(pbuff)))
