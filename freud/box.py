# Copyright (c) 2010-2023 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

r"""
The :class:`~.Box` class defines the geometry of a simulation box. The class
natively supports periodicity by providing the fundamental features for
wrapping vectors outside the box back into it.
"""

import logging
import warnings

import numpy as np

import freud.util

import freud._box

logger = logging.getLogger(__name__)


class Box:
    r"""The freud Box class for simulation boxes.

    This class defines an arbitrary triclinic geometry within which all points are confined.
    By convention, the freud Box is centered at the origin (``[0, 0, 0]``),
    with the extent in each dimension described by the half-open interval ``[-L/2, L/2)``.
    For more information, see the `documentation
    <https://freud.readthedocs.io/en/stable/gettingstarted/tutorial/periodic.html>`__
    on boxes and periodic boundary conditions.

    Also available as ``freud.Box``.

    Args:
        Lx (float, optional):
            The x-dimension length.
        Ly (float, optional):
            The y-dimension length.
        Lz (float, optional):
            The z-dimension length (Default value = 0).
        xy (float, optional):
            The xy tilt factor (Default value = 0).
        xz (float, optional):
            The xz tilt factor (Default value = 0).
        yz (float, optional):
            The yz tilt factor (Default value = 0).
        is2D (bool, optional):
            Whether the box is 2-dimensional. Uses :code:`Lz == 0`
            if :code:`None`. (Default value = :code:`None`)
    """  # noqa: E501

    def __init__(self, Lx, Ly, Lz=0, xy=0, xz=0, yz=0, is2D=None):
        if is2D is None:
            is2D = (Lz == 0)
        if is2D:
            if not (Lx and Ly):
                raise ValueError("Lx and Ly must be nonzero for 2D boxes.")
            elif Lz != 0 or xz != 0 or yz != 0:
                warnings.warn(
                    "Specifying z-dimensions in a 2-dimensional box "
                    "has no effect!")
        else:
            if not (Lx and Ly and Lz):
                raise ValueError(
                    "Lx, Ly, and Lz must be nonzero for 3D boxes.")
        self._cpp_obj = freud._box.Box(Lx, Ly, Lz, xy, xz, yz, is2D)

    @property
    def L(self):
        r""":math:`\left(3, \right)` :class:`numpy.ndarray`: Get or set the
        box lengths along x, y, and z."""
        return np.asarray([self._cpp_obj.getLx(),
                           self._cpp_obj.getLy(),
                           self._cpp_obj.getLz()])

    @L.setter
    def L(self, value):
        try:
            if len(value) != 3:
                raise ValueError('setL must be called with a scalar or a list '
                                 'of length 3.')
        except TypeError:
            # Will fail if object has no length
            value = (value, value, value)

        if self.is2D and value[2] != 0:
            warnings.warn(
                "Specifying z-dimensions in a 2-dimensional box "
                "has no effect!")
        self._cpp_obj.setL(value[0], value[1], value[2])

    @property
    def Lx(self):
        """float: Get or set the x-dimension length."""
        return self._cpp_obj.getLx()

    @Lx.setter
    def Lx(self, value):
        self.L = [value, self.Ly, self.Lz]

    @property
    def Ly(self):
        """float: Get or set the y-dimension length."""
        return self._cpp_obj.getLy()

    @Ly.setter
    def Ly(self, value):
        self.L = [self.Lx, value, self.Lz]

    @property
    def Lz(self):
        """float: Get or set the z-dimension length."""
        return self._cpp_obj.getLz()

    @Lz.setter
    def Lz(self, value):
        self.L = [self.Lx, self.Ly, value]

    @property
    def xy(self):
        """float: Get or set the xy tilt factor."""
        return self._cpp_obj.getTiltFactorXY()

    @xy.setter
    def xy(self, value):
        self._cpp_obj.setTiltFactorXY(value)

    @property
    def xz(self):
        """float: Get or set the xz tilt factor."""
        return self._cpp_obj.getTiltFactorXZ()

    @xz.setter
    def xz(self, value):
        self._cpp_obj.setTiltFactorXZ(value)

    @property
    def yz(self):
        """float: Get or set the yz tilt factor."""
        return self._cpp_obj.getTiltFactorYZ()

    @yz.setter
    def yz(self, value):
        self._cpp_obj.setTiltFactorYZ(value)

    def __eq__(self, other):
        if type(other) != freud.box.Box:
            return False
        return self.Lx == other.Lx and self.Ly == other.Ly and self.Lz == other.Lz \
            and self.xy == other.xy and self.xz == other.xz and self.yz == other.yz \
            and self.is2D == other.is2D and self.periodic_x == other.periodic_x and \
            self.periodic_y == other.periodic_y and self.periodic_z == other.periodic_z

    @property
    def dimensions(self):
        """int: Get or set the number of dimensions (2 or 3)."""
        return 2 if self.is2D else 3

    @dimensions.setter
    def dimensions(self, value):
        assert value == 2 or value == 3
        self._cpp_obj.set2D(bool(value == 2))

    @property
    def is2D(self):
        """bool: Whether the box is 2D."""
        return self._cpp_obj.is2D()

    @property
    def L_inv(self):
        r""":math:`\left(3, \right)` :class:`numpy.ndarray`: The inverse box
        lengths."""
        result = self._cpp_obj.getLinv()
        return np.asarray(result)

    @property
    def volume(self):
        """float: The box volume (area in 2D)."""
        return self._cpp_obj.getVolume()

    def make_absolute(self, fractional_coordinates, out=None):
        r"""Convert fractional coordinates into absolute coordinates.

        Args:
            fractional_coordinates (:math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Fractional coordinate vector(s), between 0 and 1 within
                parallelepipedal box.
            out (:math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray` or :code:`None`):
                The array in which to place the absolute coordinates. It must
                be of dtype :attr:`numpy.float32`. If ``None``, this function
                will return a newly allocated array (Default value = None).

        Returns:
            :math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`:
                Absolute coordinate vector(s). If ``out`` is provided, a
                reference to it is returned.
        """  # noqa: E501
        fractions = np.asarray(fractional_coordinates).copy()
        flatten = fractions.ndim == 1
        fractions = np.atleast_2d(fractions)
        fractions = freud.util._convert_array(fractions, shape=(None, 3))
        out = freud.util._convert_array(
            out, shape=fractions.shape, allow_copy=False)

        Np = fractions.shape[0]

        self._cpp_obj.makeAbsolute(fractions, Np, out)

        return np.squeeze(out) if flatten else out

    def make_fractional(self, absolute_coordinates, out=None):
        r"""Convert absolute coordinates into fractional coordinates.

        Args:
            absolute_coordinates (:math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Absolute coordinate vector(s).
            out (:math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray` or :code:`None`):
                The array in which to place the fractional positions. It must
                be of dtype :attr:`numpy.float32`. If ``None``, this function
                will return a newly allocated array (Default value = None).

        Returns:
            :math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`:
                Fractional coordinate vector(s). If ``out`` is provided, a
                reference to it is returned.
        """  # noqa: E501
        vecs = np.asarray(absolute_coordinates).copy()
        flatten = vecs.ndim == 1
        vecs = np.atleast_2d(vecs)
        vecs = freud.util._convert_array(vecs, shape=(None, 3))
        out = freud.util._convert_array(
            out, shape=vecs.shape, allow_copy=False)

        Np = vecs.shape[0]

        self._cpp_obj.makeFractional(vecs, Np, out)

        return np.squeeze(out) if flatten else out

    def get_images(self, vecs):
        r"""Returns the images corresponding to unwrapped vectors.

        Args:
            vecs (:math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Coordinates of unwrapped vector(s).

        Returns:
            :math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`:
                Image index vector(s).
        """  # noqa: E501
        vecs = np.asarray(vecs)
        flatten = vecs.ndim == 1
        vecs = np.atleast_2d(vecs)
        vecs = freud.util._convert_array(vecs, shape=(None, 3))

        images = np.zeros(vecs.shape, dtype=np.int32)
        Np = vecs.shape[0]
        self._cpp_obj.getImages(vecs, Np, images)

        return np.squeeze(images) if flatten else images

    def get_box_vector(self, i):
        r"""Get the box vector with index :math:`i`.

        Args:
            i (unsigned int):
                Index (:math:`0 \leq i < d`) of the box vector, where
                :math:`d` is the box dimension (2 or 3).

        Returns:
            :math:`\left(3, \right)` :class:`numpy.ndarray`:
                Box vector with index :math:`i`.
        """
        return self.to_matrix()[:, i]

    @property
    def v1(self):
        """:math:`(3, )` :class:`np.ndarray`: The first box vector
        :math:`(L_x, 0, 0)`."""
        return self.get_box_vector(0)

    @property
    def v2(self):
        r""":math:`(3, )` :class:`np.ndarray`: The second box vector
        :math:`(xy \times L_y, L_y, 0)`."""
        return self.get_box_vector(1)

    @property
    def v3(self):
        r""":math:`(3, )` :class:`np.ndarray`: The third box vector
        :math:`(xz \times L_z, yz \times L_z, L_z)`."""
        return self.get_box_vector(2)

    def wrap(self, vecs, out=None):
        r"""Wrap an array of vectors into the box, using periodic boundaries.

        .. note:: Since the origin of the box is in the center, wrapping is
                  equivalent to applying the minimum image convention to the
                  input vectors.

        Args:
            vecs (:math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Unwrapped vector(s).
            out (:math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray` or :code:`None`):
                The array in which to place the wrapped vectors. It must be of
                dtype :attr:`numpy.float32`. If ``None``, this function will
                return a newly allocated array (Default value = None).

        Returns:
            :math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`:
                Vector(s) wrapped into the box. If ``out`` is provided, a
                reference to it is returned.
        """  # noqa: E501
        vecs = np.asarray(vecs)
        flatten = vecs.ndim == 1
        vecs = np.atleast_2d(vecs)
        vecs = freud.util._convert_array(vecs, shape=(None, 3))
        out = freud.util._convert_array(
            out, shape=vecs.shape, allow_copy=False)

        Np = vecs.shape[0]
        self._cpp_obj.wrap(vecs, Np, out)

        return np.squeeze(out) if flatten else out

    def unwrap(self, vecs, imgs, out=None):
        r"""Unwrap an array of vectors inside the box back into real space,
        using an array of image indices that determine how many times to unwrap
        in each dimension.

        Args:
            vecs (:math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Vector(s) to be unwrapped.
            imgs (:math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Image indices for vector(s).
            out (:math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray` or :code:`None`):
                The array in which to place the unwrapped vectors. It must be
                of dtype :attr:`numpy.float32`. If ``None``, this function
                will return a newly allocated array (Default value = None).

        Returns:
            :math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`:
                Unwrapped vector(s). If ``out`` is provided, a reference to it is
                returned.
        """  # noqa: E501
        vecs = np.asarray(vecs)
        flatten = vecs.ndim == 1
        vecs = np.atleast_2d(vecs)
        imgs = np.atleast_2d(imgs)
        if vecs.shape[0] != imgs.shape[0]:
            # Broadcasts (1, 3) to (N, 3) for both arrays
            vecs, imgs = np.broadcast_arrays(vecs, imgs)
        vecs = freud.util._convert_array(vecs, shape=(None, 3)).copy()
        imgs = freud.util._convert_array(
            imgs, shape=vecs.shape, dtype=np.int32)
        out = freud.util._convert_array(
            out, shape=vecs.shape, allow_copy=False)

        Np = vecs.shape[0]
        self._cpp_obj.unwrap(vecs, imgs, Np, out)

        return np.squeeze(out) if flatten else out

    def center_of_mass(self, vecs, masses=None):
        r"""Compute center of mass of an array of vectors, using periodic boundaries.

        This calculation accounts for periodic images. `This Wikipedia page
        <https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions>`__
        describes the mathematics of this method.

        Example::

            >>> import freud
            >>> import numpy as np
            >>> box = freud.Box.cube(10)
            >>> points = [[-1, -1, 0], [-1, 1, 0], [2, 0, 0]]
            >>> np.mean(points, axis=0)  # Does not account for periodic images
            array([0., 0., 0.])
            >>> box.center_of_mass(points)  # Accounts for periodic images
            array([-0.1845932,  0.       ,  0.       ])

        Args:
            vecs (:math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Vectors used to find center of mass.
            masses (:math:`\left(N,\right)` :class:`numpy.ndarray`):
                Masses corresponding to each vector, defaulting to 1 if not
                provided or :code:`None` (Default value = :code:`None`).

        Returns:
            :math:`\left(3, \right)` :class:`numpy.ndarray`:
                Center of mass.
        """  # noqa: E501
        vecs = freud.util._convert_array(vecs, shape=(None, 3))
        Np = vecs.shape[0]

        if masses is not None:
            masses = freud.util._convert_array(masses, shape=(len(vecs), ))
        else:
            masses = np.ones(Np)

        result = self._cpp_obj.centerOfMass(vecs, Np, masses)
        return np.asarray(result)

    def center(self, vecs, masses=None):
        r"""Subtract center of mass from an array of vectors, using periodic boundaries.

        This calculation accounts for periodic images. `This Wikipedia page
        <https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions>`__
        describes the mathematics of this method.

        Example::

            >>> import freud
            >>> box = freud.Box.cube(10)
            >>> points = [[-1, -1, 0], [-1, 1, 0], [2, 0, 0]]
            >>> box.center(points)
            array([[-0.8154068, -1.0000002,  0.       ],
                   [-0.8154068,  1.       ,  0.       ],
                   [ 2.1845937,  0.       ,  0.       ]], dtype=float32)

        Args:
            vecs (:math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Vectors to center.
            masses (:math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Masses corresponding to each vector, defaulting to 1 if not
                provided or :code:`None` (Default value = :code:`None`).

        Returns:
            :math:`\left(N, 3\right)` :class:`numpy.ndarray`:
                Vectors with center of mass subtracted.
        """  # noqa: E501
        vecs = freud.util._convert_array(vecs, shape=(None, 3)).copy()
        Np = vecs.shape[0]

        if masses is not None:
            masses = freud.util._convert_array(masses, shape=(len(vecs), ))
        else:
            masses = np.ones(Np)

        self._cpp_obj.center(vecs, Np, masses)
        return vecs

    def compute_distances(self, query_points, points):
        r"""Calculate distances between two sets of points, using periodic boundaries.

        Distances are calculated row-wise, i.e. ``distances[i]`` is the
        distance from ``query_points[i]`` to ``points[i]``.

        Args:
            query_points (:math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Array of query points.
            points (:math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Array of points.

        Returns:
            :math:`\left(N, \right)` :class:`numpy.ndarray`:
                Array of distances between query points and points.
        """   # noqa: E501

        query_points = freud.util._convert_array(
            np.atleast_2d(query_points), shape=(None, 3))
        points = freud.util._convert_array(
            np.atleast_2d(points), shape=(None, 3))

        n_query_points = query_points.shape[0]
        n_points = points.shape[0]
        distances = np.empty(n_query_points, dtype=np.float32)

        self._cpp_obj.computeDistances(query_points, n_query_points, points,
                                       n_points, distances)
        return np.asarray(distances)

    def compute_all_distances(self, query_points, points):
        r"""Calculate distances between all pairs of query points and points, using periodic boundaries.

        Distances are calculated pairwise, i.e. ``distances[i, j]`` is the
        distance from ``query_points[i]`` to ``points[j]``.

        Args:
            query_points (:math:`\left(N_{query\_points}, 3 \right)` :class:`numpy.ndarray`):
                Array of query points.
            points (:math:`\left(N_{points}, 3 \right)` :class:`numpy.ndarray`):
                Array of points with same length as ``query_points``.

        Returns:
            :math:`\left(N_{query\_points}, N_{points}, \right)` :class:`numpy.ndarray`:
                Array of distances between query points and points.
        """  # noqa: E501
        query_points = freud.util._convert_array(
            np.atleast_2d(query_points), shape=(None, 3))
        points = freud.util._convert_array(
            np.atleast_2d(points), shape=(None, 3))

        n_query_points = query_points.shape[0]
        n_points = points.shape[0]
        distances = np.empty([n_query_points, n_points], dtype=np.float32)

        self._cpp_obj.computeAllDistances(query_points, n_query_points,
                                          points, n_points, distances)

        return np.asarray(distances)

    def contains(self, points):
        r"""Compute a boolean array (mask) corresponding to point membership in a box.

        This calculation computes particle membership based on conventions
        defined by :class:`Box`, ignoring periodicity. This means that in a
        cubic (3D) box with dimensions ``L``, particles would be considered
        inside the box if their coordinates are between ``[-L/2, L/2]``.
        Particles laying at a coordinate such as ``[0, L, 0]`` would be
        considered outside the box.  More information about coordinate
        conventions can be found in the documentation on `Using boxes
        <https://freud.readthedocs.io/en/latest/gettingstarted/examples/module_intros/box.Box.html#Using-boxes>`__
        and `periodic boundary conditions
        <https://freud.readthedocs.io/en/latest/gettingstarted/tutorial/periodic.html#periodic-boundary-conditions>`__.

        Example::

            >>> import freud
            >>> box = freud.Box.cube(10)
            >>> points = [[-4, 0, 0], [10, 0, 0], [0, -7, 0]]
            >>> box.contains(points)
            array([ True, False, False])

        Args:
            points (:math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Array of points.

        Returns:
            :math:`\left(N, \right)` :class:`numpy.ndarray`:
                Array of booleans, where ``True`` corresponds to points within
                the box, and ``False`` corresponds to points outside the box.
        """  # noqa: E501

        points = freud.util._convert_array(
            np.atleast_2d(points), shape=(None, 3))

        n_points = points.shape[0]

        contains_mask = freud.util._convert_array(
            np.ones(n_points), dtype=bool)

        self._cpp_obj.contains(points, n_points, contains_mask)

        return np.array(contains_mask).astype(bool)

    @property
    def cubic(self):
        """bool: Whether the box is a cube."""
        return (
            not self.is2D
            and np.allclose(
                [self.Lx, self.Lx, self.Ly, self.Ly, self.Lz, self.Lz],
                [self.Ly, self.Lz, self.Lx, self.Lz, self.Lx, self.Ly],
                rtol=1e-5,
                atol=1e-5,
            )
            and np.allclose(0, [self.xy, self.yz, self.xz], rtol=1e-5, atol=1e-5)
        )

    @property
    def periodic(self):
        r""":math:`\left(3, \right)` :class:`numpy.ndarray`: Get or set the
        periodicity of the box in each dimension."""
        return np.asarray([self._cpp_obj.getPeriodicX(),
                           self._cpp_obj.getPeriodicY(),
                           self._cpp_obj.getPeriodicZ()])

    @periodic.setter
    def periodic(self, periodic):
        # Allow passing a single value
        try:
            self._cpp_obj.setPeriodic(periodic[0], periodic[1], periodic[2])
        except TypeError:
            # Allow single value to be passed for all directions
            self._cpp_obj.setPeriodic(periodic, periodic, periodic)

    @property
    def periodic_x(self):
        """bool: Get or set the periodicity of the box in x."""
        return self._cpp_obj.getPeriodicX()

    @periodic_x.setter
    def periodic_x(self, periodic):
        self._cpp_obj.setPeriodicX(periodic)

    @property
    def periodic_y(self):
        """bool: Get or set the periodicity of the box in y."""
        return self._cpp_obj.getPeriodicY()

    @periodic_y.setter
    def periodic_y(self, periodic):
        self._cpp_obj.setPeriodicY(periodic)

    @property
    def periodic_z(self):
        """bool: Get or set the periodicity of the box in z."""
        return self._cpp_obj.getPeriodicZ()

    @periodic_z.setter
    def periodic_z(self, periodic):
        self._cpp_obj.setPeriodicZ(periodic)

    def to_dict(self):
        r"""Return box as dictionary.

        Example::

            >>> box = freud.box.Box.cube(L=10)
            >>> box.to_dict()
            {'Lx': 10.0, 'Ly': 10.0, 'Lz': 10.0,
             'xy': 0.0, 'xz': 0.0, 'yz': 0.0, 'dimensions': 3}

        Returns:
          dict: Box parameters
        """
        return {
            'Lx': self.Lx,
            'Ly': self.Ly,
            'Lz': self.Lz,
            'xy': self.xy,
            'xz': self.xz,
            'yz': self.yz,
            'dimensions': self.dimensions}

    def to_matrix(self):
        r"""Returns the box matrix (3x3).

        Example::

            >>> box = freud.box.Box.cube(L=10)
            >>> box.to_matrix()
            array([[10.,  0.,  0.],
                   [ 0., 10.,  0.],
                   [ 0.,  0., 10.]])

        Returns:
            :math:`\left(3, 3\right)` :class:`numpy.ndarray`: Box matrix
        """
        return np.asarray([[self.Lx, self.xy * self.Ly, self.xz * self.Lz],
                           [0, self.Ly, self.yz * self.Lz],
                           [0, 0, self.Lz]])

    def to_box_lengths_and_angles(self):
        r"""Return the box lengths and angles.

        Returns:
            tuple:
                The box vector lengths and angles in radians
                :math:`(L_1, L_2, L_3, \alpha, \beta, \gamma)`.
        """
        alpha = np.arccos(
            (self.xy * self.xz + self.yz)
            / (np.sqrt(1 + self.xy**2) * np.sqrt(1 + self.xz**2 + self.yz**2))
        )
        beta = np.arccos(self.xz/np.sqrt(1+self.xz**2+self.yz**2))
        gamma = np.arccos(self.xy/np.sqrt(1+self.xy**2))
        L1 = self.Lx
        a2 = [self.Ly*self.xy, self.Ly, 0]
        a3 = [self.Lz*self.xz, self.Lz*self.yz, self.Lz]
        L2 = np.linalg.norm(a2)
        L3 = np.linalg.norm(a3)
        return (L1, L2, L3, alpha, beta, gamma)

    def __repr__(self):
        return ("freud.box.{cls}(Lx={Lx}, Ly={Ly}, Lz={Lz}, "
                "xy={xy}, xz={xz}, yz={yz}, "
                "is2D={is2D})").format(cls=type(self).__name__,
                                       Lx=self.Lx,
                                       Ly=self.Ly,
                                       Lz=self.Lz,
                                       xy=self.xy,
                                       xz=self.xz,
                                       yz=self.yz,
                                       is2D=self.is2D)

    def __str__(self):
        return repr(self)

    def __mul__(self, scale):
        if scale > 0:
            return self.__class__(Lx=self.Lx*scale,
                                  Ly=self.Ly*scale,
                                  Lz=self.Lz*scale,
                                  xy=self.xy, xz=self.xz, yz=self.yz,
                                  is2D=self.is2D)
        else:
            raise ValueError("Box can only be multiplied by positive values.")

    def __rmul__(self, scale):
        return self * scale

    def plot(self, title=None, ax=None, image=[0, 0, 0], *args, **kwargs):
        """Plot a :class:`~.box.Box` object.

        Args:
            title (str):
                Title of the graph. (Default value = :code:`None`).
            ax (:class:`matplotlib.axes.Axes`):
                Axes object to plot. If :code:`None`, make a new axes and
                figure object. If plotting a 3D box, the axes must be 3D.
                (Default value = :code:`None`).
            image (list):
                The periodic image location at which to draw the box (Default
                value = :code:`[0, 0, 0]`).
            *args:
                Passed on to :meth:`mpl_toolkits.mplot3d.Axes3D.plot` or
                :meth:`matplotlib.axes.Axes.plot`.
            **kwargs:
                Passed on to :meth:`mpl_toolkits.mplot3d.Axes3D.plot` or
                :meth:`matplotlib.axes.Axes.plot`.
        """
        import freud.plot
        return freud.plot.box_plot(self, title=title, ax=ax, image=image,
                                   *args, **kwargs)

    @classmethod
    def from_box(cls, box, dimensions=None):
        r"""Initialize a Box instance from a box-like object.

        Args:
            box:
                A box-like object
            dimensions (int):
                Dimensionality of the box (Default value = None)

        .. note:: Objects that can be converted to freud boxes include
                  lists like :code:`[Lx, Ly, Lz, xy, xz, yz]`,
                  dictionaries with keys
                  :code:`'Lx', 'Ly', 'Lz', 'xy', 'xz', 'yz', 'dimensions'`,
                  objects with attributes
                  :code:`Lx, Ly, Lz, xy, xz, yz, dimensions`,
                  3x3 matrices (see :meth:`~.from_matrix`),
                  or existing :class:`freud.box.Box` objects.

                  If any of :code:`Lz, xy, xz, yz` are not provided, they will
                  be set to 0.

                  If all values are provided, a triclinic box will be
                  constructed. If only :code:`Lx, Ly, Lz` are provided, an
                  orthorhombic box will be constructed. If only :code:`Lx, Ly`
                  are provided, a rectangular (2D) box will be constructed.

                  If the optional :code:`dimensions` argument is given, this
                  will be used as the box dimensionality. Otherwise, the box
                  dimensionality will be detected from the :code:`dimensions`
                  of the provided box. If no dimensions can be detected, the
                  box will be 2D if :code:`Lz == 0`, and 3D otherwise.

        Returns:
            :class:`freud.box.Box`: The resulting box object.
        """
        if np.asarray(box).shape == (3, 3):
            # Handles 3x3 matrices
            return cls.from_matrix(box, dimensions=dimensions)
        try:
            # Handles freud.box.Box and objects with attributes
            Lx = box.Lx
            Ly = box.Ly
            Lz = getattr(box, 'Lz', 0)
            xy = getattr(box, 'xy', 0)
            xz = getattr(box, 'xz', 0)
            yz = getattr(box, 'yz', 0)
            if dimensions is None:
                dimensions = getattr(box, 'dimensions', None)
            elif dimensions != getattr(box, 'dimensions', dimensions):
                raise ValueError(
                    "The provided dimensions argument conflicts with the "
                    "dimensions attribute of the provided box object.")
        except AttributeError:
            try:
                # Handle dictionary-like
                Lx = box['Lx']
                Ly = box['Ly']
                Lz = box.get('Lz', 0)
                xy = box.get('xy', 0)
                xz = box.get('xz', 0)
                yz = box.get('yz', 0)
                if dimensions is None:
                    dimensions = box.get('dimensions', None)
                else:
                    if dimensions != box.get('dimensions', dimensions):
                        raise ValueError(
                            "The provided dimensions argument conflicts with "
                            "the dimensions attribute of the provided box "
                            "object.")
            except (IndexError, KeyError, TypeError):
                if not len(box) in [2, 3, 6]:
                    raise ValueError(
                        "List-like objects must have length 2, 3, or 6 to be "
                        "converted to freud.box.Box.")
                # Handle list-like
                Lx = box[0]
                Ly = box[1]
                Lz = box[2] if len(box) > 2 else 0
                xy, xz, yz = box[3:6] if len(box) == 6 else (0, 0, 0)
        except:  # noqa
            logger.debug('Supplied box cannot be converted to type '
                         'freud.box.Box.')
            raise

        # Infer dimensions if not provided.
        if dimensions is None:
            dimensions = 2 if Lz == 0 else 3
        is2D = (dimensions == 2)
        return cls(Lx=Lx, Ly=Ly, Lz=Lz, xy=xy, xz=xz, yz=yz, is2D=is2D)

    @classmethod
    def from_matrix(cls, box_matrix, dimensions=None):
        r"""Initialize a Box instance from a box matrix.

        For more information and the source for this code, see:
        `HOOMD-blue's box documentation \
        <https://hoomd-blue.readthedocs.io/en/stable/package-hoomd.html#hoomd.Box>`_.

        Args:
            box_matrix (array-like):
                A 3x3 matrix or list of lists
            dimensions (int):
                Number of dimensions (Default value = :code:`None`)

        Returns:
            :class:`freud.box.Box`: The resulting box object.
        """
        box_matrix = np.asarray(box_matrix, dtype=np.float32)
        v0 = box_matrix[:, 0]
        v1 = box_matrix[:, 1]
        v2 = box_matrix[:, 2]
        Lx = np.sqrt(np.dot(v0, v0))
        a2x = np.dot(v0, v1) / Lx
        Ly = np.sqrt(np.dot(v1, v1) - a2x * a2x)
        xy = a2x / Ly
        v0xv1 = np.cross(v0, v1)
        v0xv1mag = np.sqrt(np.dot(v0xv1, v0xv1))
        Lz = np.dot(v2, v0xv1) / v0xv1mag
        if Lz != 0:
            a3x = np.dot(v0, v2) / Lx
            xz = a3x / Lz
            yz = (np.dot(v1, v2) - a2x * a3x) / (Ly * Lz)
        else:
            xz = yz = 0
        if dimensions is None:
            dimensions = 2 if Lz == 0 else 3
        is2D = (dimensions == 2)
        return cls(Lx=Lx, Ly=Ly, Lz=Lz,
                   xy=xy, xz=xz, yz=yz, is2D=is2D)

    @classmethod
    def cube(cls, L=None):
        r"""Construct a cubic box with equal lengths.

        Args:
            L (float): The edge length

        Returns:
            :class:`freud.box.Box`: The resulting box object.
        """
        # classmethods compiled with cython don't appear to support
        # named access to positional arguments, so we keep this to
        # recover the behavior
        if L is None:
            raise TypeError("cube() missing 1 required positional argument: L")
        return cls(Lx=L, Ly=L, Lz=L, xy=0, xz=0, yz=0, is2D=False)

    @classmethod
    def square(cls, L=None):
        r"""Construct a 2-dimensional (square) box with equal lengths.

        Args:
            L (float): The edge length

        Returns:
            :class:`freud.box.Box`: The resulting box object.
        """
        # classmethods compiled with cython don't appear to support
        # named access to positional arguments, so we keep this to
        # recover the behavior
        if L is None:
            raise TypeError("square() missing 1 required "
                            "positional argument: L")
        return cls(Lx=L, Ly=L, Lz=0, xy=0, xz=0, yz=0, is2D=True)

    @classmethod
    def from_box_lengths_and_angles(
        cls, L1, L2, L3, alpha, beta, gamma, dimensions=None,
    ):
        r"""Construct a box from lengths and angles (in radians).

        All the angles provided must be between 0 and :math:`\pi`.

        Args:
            L1 (float):
                The length of the first lattice vector.
            L2 (float):
                The length of the second lattice vector.
            L3 (float):
                The length of the third lattice vector.
            alpha (float):
                The angle between second and third lattice vector in radians (must be
                between 0 and :math:`\pi`).
            beta (float):
                The angle between first and third lattice vector in radians (must be
                between 0 and :math:`\pi`).
            gamma (float):
                The angle between the first and second lattice vector in radians (must
                be between 0 and :math:`\pi`).
            dimensions (int):
                The number of dimensions (Default value = :code:`None`).

        Returns:
            :class:`freud.box.Box`: The resulting box object.
        """
        if not 0 < alpha < np.pi:
            raise ValueError("alpha must be between 0 and pi.")
        if not 0 < beta < np.pi:
            raise ValueError("beta must be between 0 and pi.")
        if not 0 < gamma < np.pi:
            raise ValueError("gamma must be between 0 and pi.")
        a1 = np.array([L1, 0, 0])
        a2 = np.array([L2 * np.cos(gamma), L2 * np.sin(gamma), 0])
        a3x = np.cos(beta)
        a3y = (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        under_sqrt = 1 - a3x**2 - a3y**2
        if under_sqrt < 0:
            raise ValueError("The provided angles can not form a valid box.")
        a3z = np.sqrt(under_sqrt)
        a3 = np.array([L3 * a3x, L3 * a3y, L3 * a3z])
        if dimensions is None:
            dimensions = 2 if L3 == 0 else 3
        return cls.from_matrix(np.array([a1, a2, a3]).T, dimensions=dimensions)


def BoxFromCPP(cppbox):
    b = Box(cppbox.getLx(), cppbox.getLy(), cppbox.getLz(),
            cppbox.getTiltFactorXY(), cppbox.getTiltFactorXZ(),
            cppbox.getTiltFactorYZ(), cppbox.is2D())
    b.periodic = [cppbox.getPeriodicX(),
                  cppbox.getPeriodicY(),
                  cppbox.getPeriodicZ()]
    return b
