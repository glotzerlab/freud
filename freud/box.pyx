# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The :class:`~.Box` class defines the geometry of a simulation box. The module
natively supports periodicity by providing the fundamental features for
wrapping vectors outside the box back into it. The :class:`~.PeriodicBuffer`
class is used to replicate points across the periodic boundary to assist
analysis methods that do not recognize periodic boundary conditions or extend
beyond the limits of one periodicity of the box.
"""

import warnings
import numpy as np
import freud.common

import logging

from freud.util cimport vec3
from cython.operator cimport dereference
from libcpp cimport bool as bool_t
from cpython.object cimport Py_EQ, Py_NE

cimport freud._box
cimport numpy as np

logger = logging.getLogger(__name__)

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class Box:
    R"""The freud Box class for simulation boxes.

    The Box class is defined according to the conventions of the
    HOOMD-blue simulation software.
    For more information, please see:

        https://hoomd-blue.readthedocs.io/en/stable/box.html

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
            if not provided or :code:`None`. (Default value = :code:`None`)

    Attributes:
        L (:math:`\left(3\right)` :class:`numpy.ndarray`, settable):
            The box lengths along x, y, and z.
        Lx (float, settable):
            The x-dimension length.
        Ly (float, settable):
            The y-dimension length.
        Lz (float, settable):
            The z-dimension length.
        xy (float):
            The xy tilt factor.
        xz (float):
            The xz tilt factor.
        yz (float):
            The yz tilt factor.
        is2D (bool):
            Whether the box is 2D.
        L_inv (:math:`\left(3\right)` :class:`numpy.ndarray`):
            The inverse box lengths.
        volume (float):
            The box volume (area in 2D).
        dimensions (int, settable):
            The number of dimensions (2 or 3).
        periodic (:math:`\left(3\right)` :class:`numpy.ndarray`, settable):
            Whether or not the box is periodic in each dimension.
        periodic_x (bool, settable):
            Whether or not the box is periodic in x.
        periodic_y (bool, settable):
            Whether or not the box is periodic in y.
        periodic_z (bool, settable):
            Whether or not the box is periodic in z.
    """

    def __cinit__(self, Lx, Ly, Lz=0, xy=0, xz=0, yz=0, is2D=None):
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
        self.thisptr = new freud._box.Box(Lx, Ly, Lz, xy, xz, yz, is2D)

    def __dealloc__(self):
        del self.thisptr

    @property
    def L(self):
        cdef vec3[float] result = self.thisptr.getL()
        return np.asarray([result.x, result.y, result.z])

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
        self.thisptr.setL(value[0], value[1], value[2])

    @property
    def Lx(self):
        return self.thisptr.getLx()

    @Lx.setter
    def Lx(self, value):
        self.L = [value, self.Ly, self.Lz]

    @property
    def Ly(self):
        return self.thisptr.getLy()

    @Ly.setter
    def Ly(self, value):
        self.L = [self.Lx, value, self.Lz]

    @property
    def Lz(self):
        return self.thisptr.getLz()

    @Lz.setter
    def Lz(self, value):
        self.L = [self.Lx, self.Ly, value]

    @property
    def xy(self):
        return self.thisptr.getTiltFactorXY()

    @xy.setter
    def xy(self, value):
        self.thisptr.setTiltFactorXY(value)

    @property
    def xz(self):
        return self.thisptr.getTiltFactorXZ()

    @xz.setter
    def xz(self, value):
        self.thisptr.setTiltFactorXZ(value)

    @property
    def yz(self):
        return self.thisptr.getTiltFactorYZ()

    @yz.setter
    def yz(self, value):
        self.thisptr.setTiltFactorYZ(value)

    @property
    def dimensions(self):
        return 2 if self.is2D else 3

    @dimensions.setter
    def dimensions(self, value):
        assert value == 2 or value == 3
        self.thisptr.set2D(bool(value == 2))

    @property
    def is2D(self):
        return self.thisptr.is2D()

    @property
    def L_inv(self):
        cdef vec3[float] result = self.thisptr.getLinv()
        return np.asarray([result.x, result.y, result.z])

    @property
    def volume(self):
        return self.thisptr.getVolume()

    def make_absolute(self, fractional_coordinates):
        R"""Convert fractional coordinates into absolute coordinates.

        Args:
            fractional_coordinates (:math:`\left(3\right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Fractional coordinate vector(s), between 0 and 1 within
                parallelepipedal box.

        Returns:
            :math:`\left(3\right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`:
                Absolute coordinate vector(s).
        """  # noqa: E501
        fractions = np.asarray(fractional_coordinates)
        flatten = fractions.ndim == 1
        fractions = np.atleast_2d(fractions)
        fractions = freud.common.convert_array(fractions, shape=(None, 3))

        cdef const float[:, ::1] l_points = fractions
        cdef unsigned int Np = l_points.shape[0]
        self.thisptr.makeAbsolute(<vec3[float]*> &l_points[0, 0], Np)

        return np.squeeze(fractions) if flatten else fractions

    def make_fractional(self, absolute_coordinates):
        R"""Convert absolute coordinates into fractional coordinates.

        Args:
            absolute_coordinates (:math:`\left(3\right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Absolute coordinate vector(s).

        Returns:
            :math:`\left(3\right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`:
                Fractional coordinate vector(s).
        """  # noqa: E501
        vecs = np.asarray(absolute_coordinates)
        flatten = vecs.ndim == 1
        vecs = np.atleast_2d(vecs)
        vecs = freud.common.convert_array(vecs, shape=(None, 3))

        cdef const float[:, ::1] l_points = vecs
        cdef unsigned int Np = l_points.shape[0]
        self.thisptr.makeFractional(<vec3[float]*> &l_points[0, 0], Np)

        return np.squeeze(vecs) if flatten else vecs

    def get_images(self, vecs):
        R"""Returns the images corresponding to unwrapped vectors.

        Args:
            vecs (:math:`\left(3\right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Coordinates of unwrapped vector(s).

        Returns:
            :math:`\left(3\right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`:
                Image index vector(s).
        """  # noqa: E501
        vecs = np.asarray(vecs)
        flatten = vecs.ndim == 1
        vecs = np.atleast_2d(vecs)
        vecs = freud.common.convert_array(vecs, shape=(None, 3))

        images = np.zeros(vecs.shape, dtype=np.int32)
        cdef const float[:, ::1] l_points = vecs
        cdef const int[:, ::1] l_result = images
        cdef unsigned int Np = l_points.shape[0]
        self.thisptr.getImage(<vec3[float]*> &l_points[0, 0], Np,
                              <vec3[int]*> &l_result[0, 0])

        return np.squeeze(images) if flatten else images

    def get_lattice_vector(self, i):
        R"""Get the lattice vector with index :math:`i`.

        Args:
            i (unsigned int):
                Index (:math:`0 \leq i < d`) of the lattice vector, where
                :math:`d` is the box dimension (2 or 3).

        Returns:
            :math:`\left(3\right)` :class:`numpy.ndarray`:
                Lattice vector with index :math:`i`.
        """
        cdef vec3[float] result = self.thisptr.getLatticeVector(i)
        if self.is2D:
            result.z = 0.0
        return np.asarray([result.x, result.y, result.z])

    def wrap(self, vecs):
        R"""Wrap an array of vectors into the box, using periodic boundaries.

        .. note:: Since the origin of the box is in the center, wrapping is
                  equivalent to applying the minimum image convention to the
                  input vectors.

        Args:
            vecs (:math:`\left(3\right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Unwrapped vector(s).

        Returns:
            :math:`\left(3\right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`:
                Vector(s) wrapped into the box.
        """  # noqa: E501
        vecs = np.asarray(vecs)
        flatten = vecs.ndim == 1
        vecs = np.atleast_2d(vecs)
        vecs = freud.common.convert_array(vecs, shape=(None, 3)).copy()

        cdef const float[:, ::1] l_points = vecs
        cdef unsigned int Np = l_points.shape[0]
        self.thisptr.wrap(<vec3[float]*> &l_points[0, 0], Np)

        return np.squeeze(vecs) if flatten else vecs

    def unwrap(self, vecs, imgs):
        R"""Unwrap an array of vectors inside the box back into real space,
        using an array of image indices that determine how many times to unwrap
        in each dimension.

        Args:
            vecs (:math:`\left(3\right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Vector(s) to be unwrapped.
            imgs (:math:`\left(3\right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Image indices for vector(s).

        Returns:
            :math:`\left(3\right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`:
                Unwrapped vector(s).
        """  # noqa: E501
        vecs = np.asarray(vecs)
        flatten = vecs.ndim == 1
        vecs = np.atleast_2d(vecs)
        vecs = freud.common.convert_array(vecs, shape=(None, 3)).copy()
        imgs = np.atleast_2d(imgs)
        imgs = freud.common.convert_array(imgs, shape=vecs.shape,
                                          dtype=np.int32)

        cdef const float[:, ::1] l_points = vecs
        cdef const int[:, ::1] l_imgs = imgs
        cdef unsigned int Np = l_points.shape[0]
        self.thisptr.unwrap(<vec3[float]*> &l_points[0, 0],
                            <vec3[int]*> &l_imgs[0, 0], Np)

        return np.squeeze(vecs) if flatten else vecs

    @property
    def periodic(self):
        periodic = self.thisptr.getPeriodic()
        return np.asarray([periodic.x, periodic.y, periodic.z])

    @periodic.setter
    def periodic(self, periodic):
        # Allow passing a single value
        try:
            self.thisptr.setPeriodic(periodic[0], periodic[1], periodic[2])
        except TypeError:
            # Allow single value to be passed for all directions
            self.thisptr.setPeriodic(periodic, periodic, periodic)

    @property
    def periodic_x(self):
        return self.thisptr.getPeriodicX()

    @periodic_x.setter
    def periodic_x(self, periodic):
        self.thisptr.setPeriodicX(periodic)

    @property
    def periodic_y(self):
        return self.thisptr.getPeriodicY()

    @periodic_y.setter
    def periodic_y(self, periodic):
        self.thisptr.setPeriodicY(periodic)

    @property
    def periodic_z(self):
        return self.thisptr.getPeriodicZ()

    @periodic_z.setter
    def periodic_z(self, periodic):
        self.thisptr.setPeriodicZ(periodic)

    def to_dict(self):
        R"""Return box as dictionary.

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
        R"""Returns the box matrix (3x3).

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

    def _eq(self, other):
        return self.to_dict() == other.to_dict()

    def __richcmp__(self, other, int op):
        R"""Implement all comparisons for Cython extension classes"""
        if op == Py_EQ:
            return self._eq(other)
        if op == Py_NE:
            return not self._eq(other)
        else:
            raise NotImplementedError("This comparison is not implemented")

    def __mul__(arg1, arg2):
        # Note Cython treats __mul__ and __rmul__ as one operation, so
        # type checks are necessary.
        if isinstance(arg1, freud.box.Box):
            self = arg1
            scale = arg2
        else:
            scale = arg1
            self = arg2
        if scale > 0:
            return self.__class__(Lx=self.Lx*scale,
                                  Ly=self.Ly*scale,
                                  Lz=self.Lz*scale,
                                  xy=self.xy, xz=self.xz, yz=self.yz,
                                  is2D=self.is2D)
        else:
            raise ValueError("Box can only be multiplied by positive values.")

    @classmethod
    def from_box(cls, box, dimensions=None):
        R"""Initialize a Box instance from a box-like object.

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
                  3x3 matrices (see :meth:`~.from_matrix()`),
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
            return cls.from_matrix(box)
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
            else:
                if dimensions != getattr(box, 'dimensions', dimensions):
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
                xy, xz, yz = box[3:6] if len(box) >= 6 else (0, 0, 0)
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
        R"""Initialize a Box instance from a box matrix.

        For more information and the source for this code,
        see: https://hoomd-blue.readthedocs.io/en/stable/box.html

        Args:
            box_matrix (array-like):
                A 3x3 matrix or list of lists
            dimensions (int):
                Number of dimensions (Default value = :code:`None`)
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
        R"""Construct a cubic box with equal lengths.

        Args:
            L (float): The edge length
        """
        # classmethods compiled with cython don't appear to support
        # named access to positional arguments, so we keep this to
        # recover the behavior
        if L is None:
            raise TypeError("cube() missing 1 required positional argument: L")
        return cls(Lx=L, Ly=L, Lz=L, xy=0, xz=0, yz=0, is2D=False)

    @classmethod
    def square(cls, L=None):
        R"""Construct a 2-dimensional (square) box with equal lengths.

        Args:
            L (float): The edge length
        """
        # classmethods compiled with cython don't appear to support
        # named access to positional arguments, so we keep this to
        # recover the behavior
        if L is None:
            raise TypeError("square() missing 1 required "
                            "positional argument: L")
        return cls(Lx=L, Ly=L, Lz=0, xy=0, xz=0, yz=0, is2D=True)


cdef BoxFromCPP(const freud._box.Box & cppbox):
    return Box(cppbox.getLx(), cppbox.getLy(), cppbox.getLz(),
               cppbox.getTiltFactorXY(), cppbox.getTiltFactorXZ(),
               cppbox.getTiltFactorYZ(), cppbox.is2D())


cdef class PeriodicBuffer:
    R"""Replicate periodic images of points inside a box.

    Args:
        box (:py:class:`freud.box.Box`): Simulation box.

    Attributes:
        buffer_points (:math:`\left(N_{buffer}, 3\right)` :class:`numpy.ndarray`):
            The buffer point positions.
        buffer_ids (:math:`\left(N_{buffer}\right)` :class:`numpy.ndarray`):
            The buffer point ids.
        buffer_box (:class:`freud.box.Box`):
            The buffer box, expanded to hold the replicated points.
    """  # noqa: E501

    def __cinit__(self, box):
        cdef Box b = freud.common.convert_box(box)
        self.thisptr = new freud._box.PeriodicBuffer(dereference(b.thisptr))

    def __dealloc__(self):
        del self.thisptr

    def compute(self, points, buffer, bool_t images=False):
        R"""Compute the periodic buffer.

        Args:
            points ((:math:`N_{points}`, 3) :class:`numpy.ndarray`):
                Points used to calculate periodic buffer.
            buffer (float or list of 3 floats):
                Buffer distance for replication outside the box.
            images (bool, optional):
                If ``False``, ``buffer`` is a distance. If ``True``,
                ``buffer`` is a number of images to replicate in each
                dimension. Note that one image adds half of a box length to
                each side, meaning that one image doubles the box side lengths,
                two images triples the box side lengths, and so on.
                (Default value = :code:`None`).
        """
        points = freud.common.convert_array(points, shape=(None, 3))
        cdef const float[:, ::1] l_points = points
        cdef unsigned int Np = l_points.shape[0]

        cdef vec3[float] buffer_vec
        if np.ndim(buffer) == 0:
            # catches more cases than np.isscalar
            buffer_vec = vec3[float](buffer, buffer, buffer)
        elif len(buffer) == 3:
            buffer_vec = vec3[float](buffer[0], buffer[1], buffer[2])
        else:
            raise ValueError('buffer must be a scalar or have length 3.')

        self.thisptr.compute(<vec3[float]*> &l_points[0, 0], Np, buffer_vec,
                             images)
        return self

    @property
    def buffer_points(self):
        points = self.thisptr.getBufferPoints()
        return np.asarray([[p.x, p.y, p.z] for p in points])

    @property
    def buffer_ids(self):
        return np.asarray(self.thisptr.getBufferIds())

    @property
    def buffer_box(self):
        return BoxFromCPP(<freud._box.Box> self.thisptr.getBufferBox())

    def __repr__(self):
        return ("freud.box.{cls}(box={box})").format(
            cls=type(self).__name__, box=repr(self.buffer_box))

    def __str__(self):
        return repr(self)
