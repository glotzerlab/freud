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
        basis_positions ((:math:`N_{points}`, 3) :class:`numpy.ndarray`):
            The basis of the unit cell in fractional coordinates
            (Default value = ``[[0, 0, 0]]``).
    """
    def __init__(self, box, basis_positions=[[0, 0, 0]]):
        self._box = freud.box.Box.from_box(box)
        self._basis_positions = basis_positions

    def to_system(self, num_replicas=1, scale=1, sigma_noise=0, seed=None):
        """Generate a system from the unit cell.

        The box and the positions are expanded by ``scale``, and then Gaussian
        noise with standard deviation ``sigma_noise`` is added to the
        positions.  All points are wrapped back into the box before being
        returned.

        Args:
            num_replicas (:class:`tuple` or int):
                If provided as a single number, the number of replicas in all
                dimensions. If a tuple, the number of times replicated in each
                dimension. Must be of the form (nx, ny, 1) for 2D boxes
                (Default value = 1).
            scale (float):
                Factor by which scale the unit cell (Default value = 1).
            sigma_noise (float):
                The standard deviation of the normal distribution used to add
                noise to the positions in the system (Default value = 0).
            seed (int):
                If provided, used to seed the random noise generation. Not used
                unless ``sigma_noise`` > 0 (Default value = None).

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

        if any([n > 1 for n in (nx, ny, nz)]):
            pbuff = freud.locality.PeriodicBuffer()
            pbuff.compute((self.box, self.basis_positions),
                          buffer=(nx-1, ny-1, nz-1),
                          images=True)
            box = pbuff.buffer_box*scale
            positions = np.concatenate((self.basis_positions,
                                        pbuff.buffer_points))
        else:
            box = self.box*scale
            positions = self.basis_positions.copy()
        positions *= scale

        if sigma_noise != 0:
            if seed is not None:
                random_state = np.random.get_state()
                np.random.seed(seed)
            mean = [0]*3
            var = sigma_noise*sigma_noise
            cov = np.diag([var, var, var if self.dimensions == 3 else 0])
            positions += np.random.multivariate_normal(
                mean, cov, size=positions.shape[:-1])
            if seed is not None:
                np.random.set_state(random_state)

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
        """Create an fcc crystal.

        Returns:
            :class:`~.UnitCell`: An fcc unit cell.
        """
        fractions = np.array([[.5, .5, 0],
                              [.5, 0, .5],
                              [0, .5, .5],
                              [0, 0, 0]])
        return cls([1, 1, 1], fractions)

    @classmethod
    def bcc(cls):
        """Create an bcc crystal.

        Returns:
            :class:`~.UnitCell`: An bcc unit cell.
        """
        fractions = np.array([[.5, .5, .5],
                              [0, 0, 0]])
        return cls([1, 1, 1], fractions)

    @classmethod
    def sc(cls):
        """Create an sc crystal.

        Returns:
            :class:`~.UnitCell`: An sc unit cell.
        """
        fractions = np.array([[0, 0, 0]])
        return cls([1, 1, 1], fractions)

    @classmethod
    def square(cls):
        """Create a square crystal.

        Returns:
            :class:`~.UnitCell`: A square unit cell.
        """
        fractions = np.array([[0, 0, 0]])
        return cls([1, 1], fractions)
