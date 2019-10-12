# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The :class:`freud.data` module provides certain sample data sets and utility
functions that are useful for testing and examples.
"""

import numpy as np
import freud


def make_cubic(nx=1, ny=1, nz=1, fractions=np.array([0, 0, 0],
               dtype=np.float32), scale=1.0, noise=0.0):
    """Make a cubic crystal for testing.

    Args:
        nx: Number of repeats in the x direction, default is 1.
        ny: Number of repeats in the y direction, default is 1.
        nz: Number of repeats in the z direction, default is 1.
        fractions: The basis to replicate using the lattice.
        scale: Amount to scale the unit cell by (in distance units)
               (Default value = 1.0).
        noise: Apply Gaussian noise with this width to particle positions
               (Default value = 0.0).

    Returns:
        tuple (:class:`freud.box.Box`, :class:`np.ndarray`): freud Box,
            particle positions, shape=(fractions.shape[0]*nx*ny*nz, 3)
    """

    fractions = np.tile(fractions[np.newaxis, np.newaxis, np.newaxis, ...],
                        (nx, ny, nz, 1))
    fractions[..., 0] += np.arange(nx)[:, np.newaxis, np.newaxis]
    fractions[..., 1] += np.arange(ny)[np.newaxis, :, np.newaxis]
    fractions[..., 2] += np.arange(nz)[np.newaxis, np.newaxis, :]
    fractions /= [nx, ny, nz]

    box = 2*scale*np.array([nx, ny, nz], dtype=np.float32)
    positions = ((fractions - .5)*box).reshape((-1, 3))

    if noise != 0:
        positions += np.random.normal(scale=noise, size=positions.shape)

    return freud.box.Box(*box), positions


def make_fcc(nx=1, ny=1, nz=1, scale=1.0, noise=0.0):
    """Make a FCC crystal for testing.

    Args:
        nx: Number of repeats in the x direction, default is 1
        ny: Number of repeats in the y direction, default is 1
        nz: Number of repeats in the z direction, default is 1
        scale: Amount to scale the unit cell by (in distance units)
               (Default value = 1.0).
        noise: Apply Gaussian noise with this width to particle positions
               (Default value = 0.0).

    Returns:
        tuple (:class:`freud.box.Box`, :class:`np.ndarray`): freud Box,
            particle positions, shape=(4*nx*ny*nz, 3)
    """
    fractions = np.array([[.5, .5, 0],
                          [.5, 0, .5],
                          [0, .5, .5],
                          [0, 0, 0]], dtype=np.float32)
    return make_cubic(nx, ny, nz, fractions, scale, noise)


def make_bcc(nx=1, ny=1, nz=1, scale=1.0, noise=0.0):
    """Make a body-centered-cubic crystal for testing.

    Args:
        nx: Number of repeats in the x direction, default is 1
        ny: Number of repeats in the y direction, default is 1
        nz: Number of repeats in the z direction, default is 1
        scale: Amount to scale the unit cell by (in distance units)
               (Default value = 1.0).
        noise: Apply Gaussian noise with this width to particle positions
               (Default value = 0.0).

    Returns:
        tuple (:class:`freud.box.Box`, :class:`np.ndarray`): freud Box,
            particle positions, shape=(2*nx*ny*nz, 3)
    """
    fractions = np.array([[.5, .5, .5],
                          [0, 0, 0]], dtype=np.float32)
    return make_cubic(nx, ny, nz, fractions, scale, noise)


def make_sc(nx=1, ny=1, nz=1, scale=1.0, noise=0.0):
    """Make a simple cubic for testing.

    Args:
        nx: Number of repeats in the x direction, default is 1
        ny: Number of repeats in the y direction, default is 1
        nz: Number of repeats in the z direction, default is 1
        scale: Amount to scale the unit cell by (in distance units)
               (Default value = 1.0).
        noise: Apply Gaussian noise with this width to particle positions
               (Default value = 0.0).

    Returns:
        tuple (py:class:`freud.box.Box`, :class:`np.ndarray`): freud Box,
            particle positions, shape=(nx*ny*nz, 3)
    """
    fractions = np.array([[0, 0, 0]], dtype=np.float32)
    return make_cubic(nx, ny, nz, fractions, scale, noise)


def make_square(nx=1, ny=1, fractions=np.array([[0, 0, 0]], dtype=np.float32),
                scale=1.0, noise=0.0):
    """Make a square crystal for testing

    Args:
        nx:
            Number of repeats in the x direction, default is 1
        ny:
            Number of repeats in the y direction, default is 1
        fractions:
            The basis to replicate using the lattice.
        scale:
            Amount to scale the unit cell by (in distance units), default is
            1.0
        noise:
            Apply Gaussian noise with this width to particle positions (Default
            value = 0.0)

    Returns:
        tuple (py:class:`freud.box.Box`, :class:`np.ndarray`):
            freud Box, particle positions, shape=(nx*ny*nz, 3)
    """
    box, positions = make_cubic(nx, ny, 1, fractions, scale, noise)
    box.dimensions = 2
    positions[:, 2] = 0

    return box, positions
