# Copyright (c) 2010-2020 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The :class:`freud.data` module provides certain sample data sets and utility
functions that are useful for testing and examples.

.. rubric:: Stability

:mod:`freud.data` is **unstable**. When upgrading from version 2.x to 2.y (y >
x), existing freud scripts may need to be updated. The API will be finalized in
a future release.
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
        basis_positions ((:math:`N_{points}`, 3) :class:`numpy.ndarray`):
            The basis of the unit cell in fractional coordinates
            (Default value = ``[[0, 0, 0]]``).
    """
    def __init__(self, box, basis_positions=[[0, 0, 0]]):
        self._box = freud.box.Box.from_box(box)
        self._basis_positions = basis_positions

    def generate_system(self, num_replicas=1, scale=1, sigma_noise=0,
                        seed=None):
        """Generate a system from the unit cell.

        The box and the positions are expanded by ``scale``, and then Gaussian
        noise with standard deviation ``sigma_noise`` is added to the
        positions.  All points are wrapped back into the box before being
        returned.

        Args:
            num_replicas (:class:`tuple` or int):
                If provided as a single number, the number of replicas in all
                dimensions. If a tuple, the number of times replicated in each
                dimension. Must be of the form ``(nx, ny, 1)`` for 2D boxes
                (Default value = 1).
            scale (float):
                Factor by which to scale the unit cell (Default value = 1).
            sigma_noise (float):
                The standard deviation of the normal distribution used to add
                noise to the positions in the system (Default value = 0).
            seed (int):
                If provided, used to seed the random noise generation. Not used
                unless ``sigma_noise`` > 0 (Default value = :code:`None`).

        Returns:
            tuple (:class:`freud.box.Box`, :class:`np.ndarray`):
                A system-like object (see
                :class:`~freud.locality.NeighborQuery.from_system`).
        """
        try:
            nx, ny, nz = num_replicas
        except TypeError:
            nx = ny = num_replicas
            nz = 1 if self.box.is2D else num_replicas

        if not all((int(n) == n and n > 0 for n in (nx, ny, nz))):
            raise ValueError("The number of replicas must be a positive "
                             "integer in each dimension.")

        if self.box.is2D and nz != 1:
            raise ValueError("The number of replicas in z must be 1 for a "
                             "2D unit cell.")

        if any([n > 1 for n in (nx, ny, nz)]):
            pbuff = freud.locality.PeriodicBuffer()
            abs_positions = self.box.make_absolute(self.basis_positions)
            pbuff.compute((self.box, abs_positions),
                          buffer=(nx-1, ny-1, nz-1),
                          images=True)
            box = pbuff.buffer_box*scale
            positions = np.concatenate((abs_positions, pbuff.buffer_points))
        else:
            box = self.box*scale
            positions = self.box.make_absolute(self.basis_positions)

        # Even numbers of repeats shift the box by L/2
        shift_vec = (np.array([nx, ny, nz]) + 1) % 2
        positions += shift_vec * self.box.make_absolute([1, 1, 1])
        positions *= scale
        positions = box.wrap(positions)

        if sigma_noise != 0:
            rs = np.random.RandomState(seed)
            mean = [0]*3
            var = sigma_noise*sigma_noise
            cov = np.diag([var, var, var if self.dimensions == 3 else 0])
            positions += rs.multivariate_normal(
                mean, cov, size=positions.shape[:-1])

        positions = box.wrap(positions)
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
        return self.box.to_matrix()

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
        """Create a face-centered cubic (fcc) unit cell.

        Returns:
            :class:`~.UnitCell`: A face-centered cubic unit cell.
        """
        fractions = np.array([[.5, .5, 0],
                              [.5, 0, .5],
                              [0, .5, .5],
                              [0, 0, 0]])
        return cls([1, 1, 1], fractions)

    @classmethod
    def bcc(cls):
        """Create a body-centered cubic (bcc) unit cell.

        Returns:
            :class:`~.UnitCell`: A body-centered cubic unit cell.
        """
        fractions = np.array([[.5, .5, .5],
                              [0, 0, 0]])
        return cls([1, 1, 1], fractions)

    @classmethod
    def sc(cls):
        """Create a simple cubic (sc) unit cell.

        Returns:
            :class:`~.UnitCell`: A simple cubic unit cell.
        """
        fractions = np.array([[0, 0, 0]])
        return cls([1, 1, 1], fractions)

    @classmethod
    def square(cls):
        """Create a square unit cell.

        Returns:
            :class:`~.UnitCell`: A square unit cell.
        """
        fractions = np.array([[0, 0, 0]])
        return cls([1, 1], fractions)

    @classmethod
    def hex(cls):
        """Create a hexagonal unit cell.

        Returns:
            :class:`~.UnitCell`: A hexagonal unit cell.
        """
        fractions = np.array([[0, 0, 0], [0.5, 0.5, 0]])
        return cls([1, np.sqrt(3)], fractions)


def make_random_system(box_size, num_points, is2D=False, seed=None):
    R"""Helper function to make random points with a cubic or square box.

    Args:
        box_size (float): Size of box.
        num_points (int): Number of points.
        is2D (bool): If true, creates a 2D system.
            (Default value = :code:`False`).
        seed (int): Random seed to use. (Default value = :code:`None`).

    Returns:
        tuple (:class:`freud.box.Box`, :math:`\left(num\_points, 3\right)` :class:`numpy.ndarray`):
            Generated box and points.
    """  # noqa: E501
    rs = np.random.RandomState(seed)

    fractional_coords = rs.random_sample((num_points, 3))

    if is2D:
        box = freud.box.Box.square(box_size)
        fractional_coords[:, 2] = 0
    else:
        box = freud.box.Box.cube(box_size)

    points = box.make_absolute(fractional_coords)

    return box, points
