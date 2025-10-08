# Copyright (c) 2010-2025 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from SphereVoxelization_fft import compute_2d, compute_3d

import freud

matplotlib.use("agg")


class TestSphereVoxelization:
    def test_random_points_2d(self):
        width = 100
        r_max = 10.0
        num_points = 10
        box_size = r_max * 10
        box, points = freud.data.make_random_system(box_size, num_points, is2D=True, seed=1)
        for w in (width, (width, width), [width, width]):
            vox = freud.density.SphereVoxelization(w, r_max)

            # Test access
            with pytest.raises(AttributeError):
                vox.box
            with pytest.raises(AttributeError):
                vox.voxels

            vox.compute(system=(box, points))

            # Test access
            vox.box
            vox.voxels

            # Verify the output dimensions are correct
            assert vox.voxels.shape == (width, width)
            assert np.prod(vox.voxels.shape) == np.prod(vox.width)

            # Verify the calculation is correct
            # here we assert that the calculations (from two different methods)
            # are the same up to rounding error
            fft_vox = compute_2d(box_size, width, points, r_max)
            num_same = len(
                np.where(np.isclose(vox.voxels - fft_vox, np.zeros(fft_vox.shape)))[0]
            )
            total_num = np.prod(fft_vox.shape)
            assert num_same / total_num > 0.95

            # Verify that the voxels are all 1's and 0's
            num_zeros = len(
                np.where(np.isclose(vox.voxels, np.zeros(vox.voxels.shape)))[0]
            )
            num_ones = len(
                np.where(np.isclose(vox.voxels, np.ones(vox.voxels.shape)))[0]
            )
            assert num_zeros > 0
            assert num_ones > 0
            assert num_zeros + num_ones == np.prod(vox.voxels.shape)

    def test_random_points_3d(self):
        width = 100
        r_max = 10.0
        num_points = 10
        box_size = r_max * 10
        box, points = freud.data.make_random_system(box_size, num_points, is2D=False, seed=1)
        for w in (width, (width, width, width), [width, width, width]):
            vox = freud.density.SphereVoxelization(w, r_max)

            # Test access
            with pytest.raises(AttributeError):
                vox.box
            with pytest.raises(AttributeError):
                vox.voxels

            vox.compute(system=(box, points))

            # Test access
            vox.box
            vox.voxels

            # Verify the output dimensions are correct
            assert vox.voxels.shape == (width, width, width)

            # Verify the calculation is correct
            # here we assert that the calculations (from two different methods)
            # are the same up to rounding error
            fft_vox = compute_3d(box_size, width, points, r_max)
            num_same = len(
                np.where(np.isclose(vox.voxels - fft_vox, np.zeros(fft_vox.shape)))[0]
            )
            total_num = np.prod(fft_vox.shape)
            assert num_same / total_num > 0.95

            # Verify that the voxels are all 1's and 0's
            num_zeros = len(
                np.where(np.isclose(vox.voxels, np.zeros(vox.voxels.shape)))[0]
            )
            num_ones = len(
                np.where(np.isclose(vox.voxels, np.ones(vox.voxels.shape)))[0]
            )
            assert num_zeros > 0
            assert num_ones > 0
            assert num_zeros + num_ones == np.prod(vox.voxels.shape)

    def test_change_box_dimension(self):
        width = 100
        r_max = 10.0
        num_points = 100
        box_size = r_max * 3.1

        # test that computing a 3D system after computing a 2D system will fail
        box, points = freud.data.make_random_system(box_size, num_points, is2D=True, seed=1)
        vox = freud.density.SphereVoxelization(width, r_max)
        vox.compute(system=(box, points))

        test_box, test_points = freud.data.make_random_system(
            box_size, num_points, is2D=False, seed=1
        )
        with pytest.raises(ValueError):
            vox.compute((test_box, test_points))

        # test that computing a 2D system after computing a 3D system will fail
        box, points = freud.data.make_random_system(box_size, num_points, is2D=False, seed=1)
        vox = freud.density.SphereVoxelization(width, r_max)
        vox.compute(system=(box, points))

        test_box, test_points = freud.data.make_random_system(
            box_size, num_points, is2D=True, seed=1
        )
        with pytest.raises(ValueError):
            vox.compute((test_box, test_points))

    def test_repr(self):
        vox = freud.density.SphereVoxelization(100, 10.0)
        assert str(vox) == str(eval(repr(vox)))

        # Use both signatures
        vox3 = freud.density.SphereVoxelization((98, 99, 100), 10.0)
        assert str(vox3) == str(eval(repr(vox3)))

    def test_repr_png(self):
        width = 100
        r_max = 10.0
        num_points = 100
        box_size = r_max * 3.1
        box, points = freud.data.make_random_system(box_size, num_points, is2D=True, seed=1)
        vox = freud.density.SphereVoxelization(width, r_max)

        with pytest.raises(AttributeError):
            vox.plot()
        assert vox._repr_png_() is None

        vox.compute((box, points))
        vox.plot()

        vox = freud.density.SphereVoxelization(width, r_max)
        test_box = freud.box.Box.cube(box_size)
        vox.compute((test_box, points))
        vox.plot()
        assert vox._repr_png_() is None
        plt.close("all")
