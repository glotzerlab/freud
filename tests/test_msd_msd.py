# Copyright (c) 2010-2025 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import matplotlib
import numpy as np
import numpy.testing as npt
import pytest

import freud

matplotlib.use("agg")


class TestMSD:
    def test_attribute_access(self):
        positions = np.array([[[1, 0, 0]]])
        msd = freud.msd.MSD()
        with pytest.raises(AttributeError):
            msd.msd
        with pytest.raises(AttributeError):
            msd.plot()
        assert msd._repr_png_() is None

        msd.compute(positions)
        msd.msd
        msd.box
        msd._repr_png_()

    def test_MSD(self):
        """Test correct behavior for various constructor signatures"""
        positions = np.array([[[1, 0, 0]]])
        msd = freud.msd.MSD()
        msd_direct = freud.msd.MSD(mode="direct")
        assert msd.compute(positions).msd == [0]
        assert msd_direct.compute(positions).msd == [0]

        positions = positions.repeat(10, axis=0)
        npt.assert_allclose(msd.compute(positions).msd, 0, atol=1e-4)
        npt.assert_allclose(msd_direct.compute(positions).msd, 0, atol=1e-4)

        positions[:, 0, 0] = np.arange(10)
        npt.assert_allclose(msd.compute(positions).msd, np.arange(10) ** 2, atol=1e-4)
        npt.assert_allclose(msd_direct.compute(positions).msd, np.arange(10) ** 2)

        positions = positions.repeat(2, axis=1)
        positions[:, 1, :] = 0
        npt.assert_allclose(
            msd.compute(positions).msd, np.arange(10) ** 2 / 2, atol=1e-4
        )
        npt.assert_allclose(msd_direct.compute(positions).msd, np.arange(10) ** 2 / 2)

        # Test accumulation
        positions.flags["WRITEABLE"] = False
        msd.compute(positions[:, [0], :])
        msd.compute(positions[:, [1], :], reset=False)
        msd_accumulated = msd.msd.copy()
        npt.assert_allclose(msd_accumulated, msd.compute(positions).msd)

        # Test on a lot of random data against a more naive MSD calculation.
        def simple_msd(positions):
            """A naive MSD calculation, used to test."""
            msds = []

            for m in np.arange(positions.shape[0]):
                if m:
                    diffs = positions[:-m, :, :] - positions[m:, :, :]
                else:
                    diffs = np.zeros_like(positions)
                sqdist = np.square(diffs).sum(axis=2)
                msds.append(sqdist.mean(axis=0))

            return np.array(msds).mean(axis=1), np.array(msds)

        num_tests = 5
        np.random.seed(10)
        for _ in range(num_tests):
            positions = np.random.rand(10, 10, 3)
            simple, simple_particle = simple_msd(positions)
            solution = msd.compute(positions).msd
            solution_particle = msd.compute(positions).particle_msd
            npt.assert_allclose(solution, simple, atol=1e-6)
            npt.assert_allclose(solution_particle, simple_particle, atol=1e-5)

    def test_MSD_compute(self):
        n_frames = 5
        n_particles = 10

        # Random positions: shape (N frames, particles, 3)
        positions = np.random.rand(n_frames, n_particles, 3).astype(np.float32)
        images = np.zeros((n_frames, n_particles, 3), dtype=np.int32)
        box = freud.box.Box.cube(10)

        msd = freud.msd.MSD(box=box, mode="window")
        msd.compute(positions=positions, images=images)

    def test_repr(self):
        msd = freud.msd.MSD()
        assert str(msd) == str(eval(repr(msd)))
        msd2 = freud.msd.MSD(box=freud.box.Box(1, 2, 3, 4, 5, 6), mode="direct")
        assert str(msd2) == str(eval(repr(msd2)))
