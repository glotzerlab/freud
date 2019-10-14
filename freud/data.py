# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The :class:`freud.data` module provides certain sample data sets and utility
functions that are useful for testing and examples.
"""

import numpy as np
import freud


class UnitCell(object):
    """Class to represent the unit cell of a crystal structure.

    This class represents the unit cell of a crystal structure, which is
    defined by a lattice and a basis. It provides the basic attributes of the
    unit cell as well as enabling the generation of systems of points
    (optionally with some noise) from the unit cell.

    Args:
        box:
            A box-like object (see :meth:`~freud.box.Box.from_box`) containing
            the lattice vectors of the unit cell.
        basis_positions ((:math:`N_{points}`) :class:`numpy.ndarray`):
            The basis of the unit cell in fractional coordinates
            (Default value = ``[[0, 0, 0]]``.
    """
    def __init__(self, box, basis_positions=[[0, 0, 0]]):
        self._box = freud.box.Box.from_box(box)
        self._basis_positions = np.dot(basis_positions, self.box.to_matrix().T)

    def to_system(self, num_replicas=1, scale=1, sigma_noise=0):
        """Generate a system from the unit cell.

        Args:
            num_replicas (:class:`tuple` or `int`):
                If provided as a single number, the number of replicas in all
                dimensions. If a tuple, the number of times replicated in each
                dimension. Must be of the form (nx, ny, 1) for 2D boxes
                (Default value = 1).
            scale (float): Amount to scale the unit cell by (in distance units)
                (Default value = 1).
            sigma_noise (float):
                The standard deviation of the normal distribution used to add
                noise to the positions in the system (Default value = 0).

        Returns:
            tuple (:class:`freud.box.Box`, :class:`np.ndarray`):
                A system-like object (see
                :class:`~freud.locality.NeighborQuery.from_system`.).
        """
        try:
            nx, ny, nz = num_replicas
        except TypeError:
            nx = ny = num_replicas
            nz = 1 if self.box.is2D else num_replicas

        if self.box.is2D and nz != 1:
            raise ValueError("The number of replicas in z must be 1 for a "
                             "2D unit cell.")

        basis = np.tile(
            self.basis_positions[np.newaxis, np.newaxis, np.newaxis, ...],
            (nx, ny, nz, 1, 1))
        basis[..., 0] += np.arange(nx)[:, np.newaxis, np.newaxis, np.newaxis]
        basis[..., 1] += np.arange(ny)[np.newaxis, :, np.newaxis, np.newaxis]
        basis[..., 2] += np.arange(nz)[np.newaxis, np.newaxis, :, np.newaxis]

        positions = (basis - 0.5).reshape(-1, 3).dot(
            self.box.to_matrix().T)*scale
        box = freud.box.Box.from_matrix(
            self.box.to_matrix() * scale * [[nx, ny, nz]])

        if sigma_noise != 0:
            mean = [0]*3
            var = sigma_noise*sigma_noise
            cov = np.diag([var, var, var if self.dimensions == 3 else 0])
            positions += np.random.multivariate_normal(
                mean, cov, size=positions.shape[:-1])

        return box, positions

    @property
    def box(self):
        """:class:`freud.box.Box`: The box instance containing the lattice
        vectors."""
        return self._box

    @property
    def lattice_vectors(self):
        """:math:`(3, 3)` :class:`np.ndarray`: The matrix of lattice
        vectors."""
        return self._box.to_matrix()

    @property
    def basis_positions(self):
        """:math:`(N_{points}, 3)` :class:`np.ndarray`: The basis positions."""
        return self._basis_positions

    @property
    def a1(self):
        """:math:`(3, )` :class:`np.ndarray`: The first lattice vector."""
        return self.box.to_matrix()[:, 0]

    @property
    def a2(self):
        """:math:`(3, )` :class:`np.ndarray`: The second lattice vector."""
        return self.box.to_matrix()[:, 1]

    @property
    def a3(self):
        """:math:`(3, )` :class:`np.ndarray`: The third lattice vector."""
        return self.box.to_matrix()[:, 2]

    @property
    def dimensions(self):
        """int: The dimensionality of the unit cell."""
        return self.box.dimensions

    @classmethod
    def fcc(cls):
        """Create an FCC crystal.

        Returns:
            :class:`~.UnitCell`: An FCC unit cell.
        """
        fractions = np.array([[.5, .5, 0],
                              [.5, 0, .5],
                              [0, .5, .5],
                              [0, 0, 0]], dtype=np.float32)
        return cls([1, 1, 1], fractions)

    @classmethod
    def bcc(cls):
        """Create an BCC crystal.

        Returns:
            :class:`~.UnitCell`: An BCC unit cell.
        """
        fractions = np.array([[.5, .5, .5],
                              [0, 0, 0]], dtype=np.float32)
        return cls([1, 1, 1], fractions)

    @classmethod
    def sc(cls):
        """Create an SC crystal.

        Returns:
            :class:`~.UnitCell`: An SC unit cell.
        """
        fractions = np.array([[0, 0, 0]], dtype=np.float32)
        return cls([1, 1, 1], fractions)

    @classmethod
    def square(cls):
        """Create a square crystal.

        Returns:
            :class:`~.UnitCell`: A square unit cell.
        """
        fractions = np.array([[0, 0, 0]], dtype=np.float32)
        return cls([1, 1], fractions)
