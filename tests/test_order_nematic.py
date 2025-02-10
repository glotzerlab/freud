# Copyright (c) 2010-2025 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np
import numpy.testing as npt
import pytest

import freud


class TestNematicOrder:
    def test_perfect(self):
        """Test different perfectly aligned systems."""
        N = 10000
        u = [1, 0, 0]
        orientations = np.repeat(np.expand_dims(u, axis=0), repeats=N, axis=0)

        op_parallel = freud.order.Nematic()

        # Test access
        with pytest.raises(AttributeError):
            op_parallel.order
        with pytest.raises(AttributeError):
            op_parallel.director
        with pytest.raises(AttributeError):
            op_parallel.particle_tensor
        with pytest.raises(AttributeError):
            op_parallel.nematic_tensor

        op_parallel.compute(orientations)

        # Test access
        op_parallel.order
        op_parallel.director
        op_parallel.particle_tensor
        op_parallel.nematic_tensor

        assert op_parallel.order == 1
        npt.assert_equal(op_parallel.director, u)
        npt.assert_equal(op_parallel.nematic_tensor, np.diag([1, -0.5, -0.5]))
        npt.assert_equal(
            op_parallel.nematic_tensor,
            np.mean(op_parallel.particle_tensor, axis=0),
        )

        u = [0, 1, 0]
        orientations = np.repeat(np.expand_dims(u, axis=0), repeats=N, axis=0)
        op_perp = freud.order.Nematic()
        op_perp.compute(orientations)

        assert op_perp.order == 1
        npt.assert_equal(op_perp.director, u)
        npt.assert_equal(op_perp.nematic_tensor, np.diag([-0.5, 1, -0.5]))

    def test_imperfect(self):
        """Test imperfectly aligned systems.
        We add some noise to the perfect system and see if the output is close
        to the ideal case.
        """
        N = 10000
        np.random.seed(0)
        u = [1, 0, 0]

        # Generate orientations close to the u
        orientations = np.random.normal(
            np.repeat(np.expand_dims(u, axis=0), repeats=N, axis=0), 0.1
        )

        op = freud.order.Nematic()
        op.compute(orientations)

        npt.assert_allclose(op.order, 1, atol=1e-1)
        assert op.order != 1

        npt.assert_allclose(op.director, u, atol=1e-1)
        assert not np.all(op.director == u)

        npt.assert_allclose(op.nematic_tensor, np.diag([1, -0.5, -0.5]), atol=1e-1)
        assert not np.all(op.nematic_tensor == np.diag([1, -0.5, -0.5]))

        u = np.array([0, 1, 0])
        orientations = np.random.normal(
            np.repeat(np.expand_dims(u, axis=0), repeats=N, axis=0), 0.1
        )
        op_perp = freud.order.Nematic()
        op_perp.compute(orientations)

        npt.assert_allclose(op_perp.order, 1, atol=1e-1)
        assert op_perp.order != 1

        npt.assert_allclose(np.abs(op_perp.director), u, atol=1e-1)
        assert not np.all(op_perp.director == u)

        npt.assert_allclose(op_perp.nematic_tensor, np.diag([-0.5, 1, -0.5]), atol=1e-1)
        assert not np.all(op_perp.nematic_tensor == np.diag([-0.5, 1, -0.5]))

    def test_warning(self):
        """Test that supplying a zero orientation vector raises a warning."""
        N = 10000
        np.random.seed(0)
        u = [1, 0, 0]

        # Generate orientations close to the u
        orientations = np.random.normal(
            np.repeat(np.expand_dims(u, axis=0), repeats=N, axis=0), 0.1
        )

        # Change first orientation to zero vector
        orientations[0] = np.array([0, 0, 0])

        op = freud.order.Nematic()

        with pytest.warns(UserWarning):
            op.compute(orientations)

    def test_repr(self):
        op = freud.order.Nematic()
        assert str(op) == str(eval(repr(op)))
