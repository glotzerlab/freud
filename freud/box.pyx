# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The box module provides the Box class, which defines the geometry of the
simulation box. The module natively supports periodicity by providing the
fundamental features for wrapping vectors outside the box back into it.
"""

from __future__ import print_function

import warnings
import numpy as np
from collections import namedtuple
import freud.common
from freud.errors import FreudDeprecationWarning

import logging

from freud.util._VectorMath cimport vec3
from libcpp.string cimport string
from libc.string cimport memcpy
from cpython.object cimport Py_EQ, Py_NE

cimport freud._box
cimport numpy as np

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

logger = logging.getLogger(__name__)

cdef class Box:
    """The freud Box class for simulation boxes.

    .. moduleauthor:: Richmond Newman <newmanrs@umich.edu>
    .. moduleauthor:: Carl Simon Adorf <csadorf@umich.edu>
    .. moduleauthor:: Bradley Dice <bdice@bradleydice.com>

    .. versionchanged:: 0.7.0
       Added box periodicity interface

    The Box class is defined according to the conventions of the
    HOOMD-blue simulation software.
    For more information, please see:

        http://hoomd-blue.readthedocs.io/en/stable/box.html

    Args:
        Lx (float):
            Length of side x.
        Ly (float):
            Length of side y.
        Lz (float):
            Length of side z.
        xy (float):
            Tilt of xy plane.
        xz (float):
            Tilt of xz plane.
        yz (float):
            Tilt of yz plane.
        is2D(bool):
            Specify that this box is 2-dimensional, default is 3-dimensional.

    Attributes:
        xy (float):
            The xy tilt factor.
        xz (float):
            The xz tilt factor.
        yz (float):
            The yz tilt factor.
        L (tuple, settable):
            The box lengths
        Lx (tuple, settable):
            The x-dimension length.
        Ly (tuple, settable):
            The y-dimension length.
        Lz (tuple, settable):
            The z-dimension length.
        Linv (tuple):
            The inverse box lengths.
        volume (float):
            The box volume (area in 2D).
        dimensions (int, settable):
            The number of dimensions (2 or 3).
        periodic (list, settable):
            Whether or not the box is periodic.
        periodic_x (bool, settable):
            Whether or not the box is periodic in x.
        periodic_y (bool, settable):
            Whether or not the box is periodic in y.
        periodic_z (bool, settable):
            Whether or not the box is periodic in z.
    """
    def __cinit__(self, Lx=None, Ly=None, Lz=None, xy=None, xz=None, yz=None,
                  is2D=None):
        if Lx is None:
            Lx = 0
        if Ly is None:
            Ly = 0
        if Lz is None:
            Lz = 0
        if xy is None:
            xy = 0
        if xz is None:
            xz = 0
        if yz is None:
            yz = 0
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
                raise ValueError("Lx, Ly and Lz must be nonzero for 2D boxes.")
        self.thisptr = new freud._box.Box(Lx, Ly, Lz, xy, xz, yz, is2D)

    def __dealloc__(self):
        del self.thisptr

    @property
    def L(self):
        cdef vec3[float] result = self.thisptr.getL()
        return (result.x, result.y, result.z)

    @L.setter
    def L(self, value):
        try:
            if len(value) != 3:
                raise ValueError('setL must be called with a scalar or a list '
                                 'of length 3.')
        except TypeError:
            # Will fail if object has no length
            value = (value, value, value)

        if self.is2D() and value[2] != 0:
            warnings.warn(
                "Specifying z-dimensions in a 2-dimensional box "
                "has no effect!")
        self.thisptr.setL(value[0], value[1], value[2])

    def getL(self):
        warnings.warn("The getL function is deprecated in favor "
                      "of the L class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.L

    def setL(self, L):
        warnings.warn("The setL function is deprecated in favor "
                      "of setting the L class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        self.L = L

    @property
    def Lx(self):
        return self.thisptr.getLx()

    @Lx.setter
    def Lx(self, value):
        self.L = [value, self.Ly, self.Lz]

    def getLx(self):
        warnings.warn("The getLx function is deprecated in favor "
                      "of the Lx class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.Lx

    @property
    def Ly(self):
        return self.thisptr.getLy()

    @Ly.setter
    def Ly(self, value):
        self.L = [self.Lx, value, self.Lz]

    def getLy(self):
        warnings.warn("The getLy function is deprecated in favor "
                      "of the Ly class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.Ly

    @property
    def Lz(self):
        return self.thisptr.getLz()

    @Lz.setter
    def Lz(self, value):
        self.L = [self.Lx, self.Ly, value]

    def getLz(self):
        warnings.warn("The getLz function is deprecated in favor "
                      "of the Lz class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.Lz

    @property
    def xy(self):
        return self.thisptr.getTiltFactorXY()

    def getTiltFactorXY(self):
        warnings.warn("The getTiltFactorXY function is deprecated in favor "
                      "of the xy class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.xy

    @property
    def xz(self):
        return self.thisptr.getTiltFactorXZ()

    def getTiltFactorXZ(self):
        warnings.warn("The getTiltFactorXZ function is deprecated in favor "
                      "of the xz class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.xz

    @property
    def yz(self):
        return self.thisptr.getTiltFactorYZ()

    def getTiltFactorYZ(self):
        warnings.warn("The getTiltFactorYZ function is deprecated in favor "
                      "of the yz class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.yz

    @property
    def dimensions(self):
        return 2 if self.is2D() else 3

    @dimensions.setter
    def dimensions(self, value):
        assert value == 2 or value == 3
        self.thisptr.set2D(bool(value == 2))

    def is2D(self):
        """Return if box is 2D (True) or 3D (False).

        Returns:
            bool: True if 2D, False if 3D.
        """
        return self.thisptr.is2D()

    def set2D(self, val):
        warnings.warn("The set2D function is deprecated in favor "
                      "of setting the dimensions class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        self.dimensions = 2 if val else 3

    @property
    def Linv(self):
        cdef vec3[float] result = self.thisptr.getLinv()
        return (result.x, result.y, result.z)

    def getLinv(self):
        warnings.warn("The getLinv function is deprecated in favor "
                      "of the Linv class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.Linv

    @property
    def volume(self):
        return self.thisptr.getVolume()

    def getVolume(self):
        warnings.warn("The getVolume function is deprecated in favor "
                      "of the volume class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.volume

    def getCoordinates(self, f):
        warnings.warn("The getCoordinates function is deprecated in favor "
                      "of the makeCoordinates function and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.makeCoordinates(f)

    def makeCoordinates(self, f):
        """Convert fractional coordinates into real coordinates.

        Args:
            f (:math:`\\left(3\\right)` :class:`numpy.ndarray`):
                Fractional coordinates :math:`\\left(x, y, z\\right)` between
                0 and 1 within parallelepipedal box.

        Returns:
            list[float, float, float]:
                Vector of real coordinates :math:`\\left(x, y, z\\right)`.
        """
        cdef np.ndarray[float, ndim=1] l_vec = freud.common.convert_array(
            f, 1, dtype=np.float32, contiguous=True)
        cdef vec3[float] result = self.thisptr.makeCoordinates(
            <const vec3[float]&> l_vec[0])
        return [result.x, result.y, result.z]

    def makeFraction(self, vec):
        """Convert real coordinates into fractional coordinates.

        Args:
            vec (:math:`\\left(3\\right)` :class:`numpy.ndarray`):
                Real coordinates within parallelepipedal box.

        Returns:
            list[float, float, float]:
                A fractional coordinate vector.
        """
        cdef np.ndarray[float, ndim=1] l_vec = freud.common.convert_array(
            vec, 1, dtype=np.float32, contiguous=True)
        cdef vec3[float] result = self.thisptr.makeFraction(
            <const vec3[float]&> l_vec[0])
        return [result.x, result.y, result.z]

    def getImage(self, vec):
        """Returns the image corresponding to a wrapped vector.

        .. versionadded:: 0.8

        Args:
            vec (:math:`\\left(3\\right)` :class:`numpy.ndarray`):
                Coordinates of unwrapped vector.

        Returns:
            :math:`\\left(3\\right)` :class:`numpy.ndarray`:
                Image index vector.
        """
        cdef np.ndarray[float, ndim=1] l_vec = freud.common.convert_array(
            vec, 1, dtype=np.float32, contiguous=True)
        cdef vec3[int] result = self.thisptr.getImage(
            <const vec3[float]&> l_vec[0])
        return [result.x, result.y, result.z]

    def getLatticeVector(self, i):
        """Get the lattice vector with index :math:`i`.

        Args:
            i (unsigned int):
                Index (:math:`0 \\leq i < d`) of the lattice vector, where \
                :math:`d` is the box dimension (2 or 3).

        Returns:
            list[float, float, float]: Lattice vector with index :math:`i`.
        """
        cdef unsigned int index = i
        cdef vec3[float] result = self.thisptr.getLatticeVector(i)
        if self.thisptr.is2D():
            result.z = 0.0
        return [result.x, result.y, result.z]

    def wrap(self, vecs):
        """Wrap a given array of vectors from real space into the box, using
        the periodic boundaries.

        .. note:: Since the origin of the box is in the center, wrapping is
                  equivalent to applying the minimum image convention to the
                  input vectors.

        Args:
            vecs (:math:`\\left(3\\right)` or :math:`\\left(N, 3\\right)` \
            :class:`numpy.ndarray`):
                Single vector or array of :math:`N` vectors. The vectors are
                altered in place and returned.

        Returns:
            :math:`\\left(3\\right)` or :math:`\\left(N, 3\\right)` \
            :class:`numpy.ndarray`:
                Vectors wrapped into the box.
        """
        vecs = np.asarray(vecs)
        if vecs.ndim > 2 or vecs.shape[-1] != 3:
            raise ValueError(
                "Invalid dimensions for vecs given to box.wrap. "
                "Valid input is an array of shape (3,) or (N,3).")

        vecs = freud.common.convert_array(
            vecs, vecs.ndim, dtype=np.float32, contiguous=True)

        if vecs.ndim == 1:
            # only one vector to wrap
            vecs[:] = self._wrap(vecs)
        elif vecs.ndim == 2:
            for i, vec in enumerate(vecs):
                vecs[i] = self._wrap(vec)
        return vecs

    def _wrap(self, vec):
        """Wrap a single vector."""
        cdef np.ndarray[float, ndim=1] l_vec = vec
        cdef vec3[float] result = self.thisptr.wrap(<vec3[float]&> l_vec[0])
        return (result.x, result.y, result.z)

    def unwrap(self, vecs, imgs):
        """Unwrap a given array of vectors inside the box back into real space,
        using an array of image indices that determine how many times to
        unwrap in each dimension.

        Args:
            vecs (:math:`\\left(3\\right)` or :math:`\\left(N, 3\\right)` \
            :class:`numpy.ndarray`):
                Single vector or array of :math:`N` vectors. The vectors are
                modified in place.
            imgs (:math:`\\left(3\\right)` or :math:`\\left(N, 3\\right)` \
            :class:`numpy.ndarray`):
                Single image index or array of :math:`N` image indices.

        Returns:
            :math:`\\left(3\\right)` or :math:`\\left(N, 3\\right)` \
            :class:`numpy.ndarray`:
                Vectors unwrapped by the image indices provided.
        """
        vecs = np.asarray(vecs)
        imgs = np.asarray(imgs)
        if vecs.shape != imgs.shape:
            raise ValueError("imgs dimensions do not match vecs dimensions.")

        if vecs.ndim > 2 or vecs.shape[-1] != 3:
            raise ValueError(
                "Invalid dimensions for vecs given to box.unwrap. "
                "Valid input is an array of shape (3,) or (N,3).")

        vecs = freud.common.convert_array(
            vecs, vecs.ndim, dtype=np.float32, contiguous=True)
        imgs = freud.common.convert_array(
            imgs, vecs.ndim, dtype=np.int32, contiguous=True)

        if vecs.ndim == 1:
            # only one vector to unwrap
            vecs = self._unwrap(vecs, imgs)
        elif vecs.ndim == 2:
            for i, (vec, img) in enumerate(zip(vecs, imgs)):
                vecs[i] = self._unwrap(vec, img)
        return vecs

    def _unwrap(self, vec, img):
        """Unwrap a single vector."""
        cdef np.ndarray[float, ndim=1] l_vec = vec
        cdef np.ndarray[int, ndim=1] l_img = img
        cdef vec3[float] result = self.thisptr.unwrap(
            <vec3[float]&> l_vec[0], <vec3[int]&> l_img[0])
        return [result.x, result.y, result.z]

    def getPeriodic(self):
        warnings.warn("The getPeriodic function is deprecated in favor "
                      "of the periodic class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        periodic = self.thisptr.getPeriodic()
        return [periodic.x, periodic.y, periodic.z]

    def setPeriodic(self, x, y, z):
        warnings.warn("The setPeriodic function is deprecated in favor "
                      "of setting the periodic class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        self.thisptr.setPeriodic(x, y, z)

    @property
    def periodic(self):
        return self.getPeriodic()

    @periodic.setter
    def periodic(self, periodic):
        # Allow passing a single value
        try:
            self.setPeriodic(periodic[0], periodic[1], periodic[2])
        except TypeError:
            # Allow single value to be passed for all directions
            self.setPeriodic(periodic, periodic, periodic)

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

    def getPeriodicX(self):
        warnings.warn("The getPeriodicX function is deprecated in favor "
                      "of setting the periodic_x class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.periodic_x

    def setPeriodicX(self, val):
        warnings.warn("The setPeriodicX function is deprecated in favor "
                      "of setting the periodic_x class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        self.periodic_x = val

    def getPeriodicY(self):
        warnings.warn("The getPeriodicY function is deprecated in favor "
                      "of setting the periodic_y class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.periodic_y

    def setPeriodicY(self, val):
        warnings.warn("The setPeriodicY function is deprecated in favor "
                      "of setting the periodic_y class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        self.periodic_y = val

    def getPeriodicZ(self):
        warnings.warn("The getPeriodicZ function is deprecated in favor "
                      "of setting the periodic_z class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.periodic_z

    def setPeriodicZ(self, val):
        warnings.warn("The setPeriodicZ function is deprecated in favor "
                      "of setting the periodic_z class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        self.periodic_z = val

    def to_dict(self):
        """Return box as dictionary.

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

    def to_tuple(self):
        """Returns the box as named tuple.

        Returns:
            namedtuple: Box parameters
        """
        tuple_type = namedtuple(
            'BoxTuple', ['Lx', 'Ly', 'Lz', 'xy', 'xz', 'yz'])
        return tuple_type(Lx=self.Lx, Ly=self.Ly, Lz=self.Lz,
                          xy=self.xy, xz=self.xz, yz=self.yz)

    def to_matrix(self):
        """Returns the box matrix (3x3).

        Returns:
            list of lists, shape 3x3: box matrix
        """
        return [[self.Lx, self.xy * self.Ly, self.xz * self.Lz],
                [0, self.Ly, self.yz * self.Lz],
                [0, 0, self.Lz]]

    def __str__(self):
        return ("{cls}(Lx={Lx}, Ly={Ly}, Lz={Lz}, xy={xy}, "
                "xz={xz}, yz={yz}, dimensions={dimensions})").format(
                    cls=type(self).__name__, **self.to_dict())

    def _eq(self, other):
        return self.to_dict() == other.to_dict()

    def __richcmp__(self, other, int op):
        """Implement all comparisons for Cython extension classes"""
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
                                  is2D=self.is2D())
        else:
            raise ValueError("Box can only be multiplied by positive values.")

    @classmethod
    def from_box(cls, box, dimensions=None):
        """Initialize a box instance from a box-like object.

        Args:
            box:
                A box-like object
            dimensions (int):
                Dimensionality of the box (Default value = None)

        .. note:: Objects that can be converted to freud boxes include
                  lists like :code:`[Lx, Ly, Lz, xy, xz, yz]`,
                  dictionaries with keys
                  :code:`'Lx', 'Ly', 'Lz', 'xy', 'xz', 'yz', 'dimensions'`,
                  namedtuples with properties
                  :code:`Lx, Ly, Lz, xy, xz, yz, dimensions`,
                  3x3 matrices (see :py:meth:`~.from_matrix()`),
                  or existing :py:class:`freud.box.Box` objects.

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
            :class:`freud.box:Box`: The resulting box object.
        """
        if np.asarray(box).shape == (3, 3):
            # Handles 3x3 matrices
            return cls.from_matrix(box)
        try:
            # Handles freud.box.Box and namedtuple
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
            except (KeyError, TypeError):
                if not len(box) in [2, 3, 6]:
                    raise ValueError(
                        "List-like objects must have length 2, 3, or 6 to be "
                        "converted to a box")
                # Handle list-like
                Lx = box[0]
                Ly = box[1]
                Lz = box[2] if len(box) > 2 else 0
                xy, xz, yz = box[3:6] if len(box) >= 6 else (0, 0, 0)
        except:  # noqa
            logger.debug('Supplied box cannot be converted to type '
                         'freud.box.Box')
            raise

        # Infer dimensions if not provided.
        if dimensions is None:
            dimensions = 2 if Lz == 0 else 3
        is2D = (dimensions == 2)
        return cls(Lx=Lx, Ly=Ly, Lz=Lz, xy=xy, xz=xz, yz=yz, is2D=is2D)

    @classmethod
    def from_matrix(cls, boxMatrix, dimensions=None):
        """Initialize a box instance from a box matrix.

        For more information and the source for this code,
        see: http://hoomd-blue.readthedocs.io/en/stable/box.html

        Args:
            boxMatrix (array-like):
                A 3x3 matrix or list of lists
            dimensions (int):
                Number of dimensions (Default value = None)
        """
        boxMatrix = np.asarray(boxMatrix, dtype=np.float32)
        v0 = boxMatrix[:, 0]
        v1 = boxMatrix[:, 1]
        v2 = boxMatrix[:, 2]
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
        return cls(Lx=Lx, Ly=Ly, Lz=Lz,
                   xy=xy, xz=xz, yz=yz, is2D=dimensions == 2)

    @classmethod
    def cube(cls, L=None):
        """Construct a cubic box with equal lengths.

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
        """Construct a 2-dimensional (square) box with equal lengths.

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

    # Enable box to be pickled
    def __getinitargs__(self):
        return (self.getLx(), self.getLy(), self.getLz(),
                self.getTiltFactorXY(),
                self.getTiltFactorXZ(),
                self.getTiltFactorYZ(),
                self.is2D())

cdef BoxFromCPP(const freud._box.Box & cppbox):
    return Box(cppbox.getLx(), cppbox.getLy(), cppbox.getLz(),
               cppbox.getTiltFactorXY(), cppbox.getTiltFactorXZ(),
               cppbox.getTiltFactorYZ(), cppbox.is2D())
