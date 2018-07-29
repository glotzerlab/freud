# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

import warnings
import numpy as np
from collections import namedtuple
from freud.util._VectorMath cimport vec3
from libcpp.string cimport string
from libc.string cimport memcpy
from cpython.object cimport Py_EQ, Py_NE
cimport freud._box as box
cimport numpy as np

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

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
        Lx (float): Length of side x
        Ly (float): Length of side y
        Lz (float): Length of side z
        xy (float): Tilt of xy plane
        xz (float): Tilt of xz plane
        yz (float): Tilt of yz plane
        is2D(bool): Specify that this box is 2-dimensional,
            default is 3-dimensional.

    Attributes:
        xy (float): The xy tilt factor
        xz (float): The xz tilt factor
        yz (float): The yz tilt factor
        L (tuple, settable): The box lengths
        Lx (tuple, settable): The x-dimension length
        Ly (tuple, settable): The y-dimension length
        Lz (tuple, settable): The z-dimension length
        Linv (tuple): The inverse box lengths
        volume (float): The box volume (area in 2D)
        dimensions (int, settable): The number of dimensions (2 or 3)
        periodic (list, settable): Whether or not the box is periodic
    """
    cdef box.Box * thisptr

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
            is2D = False
        if is2D and (Lz != 0 or xz != 0 or yz != 0):
            warnings.warn(
                "Specifying z-dimensions in a 2-dimensional box "
                "has no effect!")
        self.thisptr = new box.Box(Lx, Ly, Lz, xy, xz, yz, is2D)

    def __dealloc__(self):
        del self.thisptr

    def getL(self):
        """Return the lengths of the box as a tuple (x, y, z).

        Returns:
            (float, float, float): Dimensions of the box as (x, y, z).
        """
        cdef vec3[float] result = self.thisptr.getL()
        return (result.x, result.y, result.z)

    def setL(self, L):
        """Set all side lengths of box to L.

        Args:
            L (float): Side length of box.
        """
        try:
            len(L)
        except TypeError:
            L = (L, L, L)

        if len(L) != 3:
            raise TypeError('Could not setL({})'.format(L))

        if self.is2D() and L[2] != 0:
            warnings.warn(
                "Specifying z-dimensions in a 2-dimensional box "
                "has no effect!")
        self.thisptr.setL(L[0], L[1], L[2])

    def getLx(self):
        """Length of the x-dimension of the box.

        Returns:
            float: This box's x-dimension length.
        """
        return self.thisptr.getLx()

    def getLy(self):
        """Length of the y-dimension of the box.

        Returns:
            float: This box's y-dimension length.
        """
        return self.thisptr.getLy()

    def getLz(self):
        """Length of the z-dimension of the box.

        Returns:
            float: This box's z-dimension length.
        """
        return self.thisptr.getLz()

    def getTiltFactorXY(self):
        """Return the tilt factor xy.

        Returns:
            float: This box's xy tilt factor.
        """
        return self.thisptr.getTiltFactorXY()

    @property
    def xy(self):
        return self.getTiltFactorXY()

    def getTiltFactorXZ(self):
        """Return the tilt factor xz.

        Returns:
            float: This box's xz tilt factor.
        """
        return self.thisptr.getTiltFactorXZ()

    @property
    def xz(self):
        return self.getTiltFactorXZ()

    def getTiltFactorYZ(self):
        """Return the tilt factor yz.

        Returns:
            float: This box's yz tilt factor.
        """
        return self.thisptr.getTiltFactorYZ()

    @property
    def yz(self):
        return self.getTiltFactorYZ()

    def is2D(self):
        """Return if box is 2D (True) or 3D (False).

        Returns:
            bool: True if 2D, False if 3D.
        """
        return self.thisptr.is2D()

    def set2D(self, val):
        """Set the dimensionality to 2D (True) or 3D (False).

        Args:
            val (bool): 2D=True, 3D=False.
        """
        self.thisptr.set2D(bool(val))

    def getLinv(self):
        """Return the inverse lengths of the box (1/Lx, 1/Ly, 1/Lz).

        Returns:
            (float, float, float): dimensions of the box as (1/Lx, 1/Ly, 1/Lz).
        """
        cdef vec3[float] result = self.thisptr.getLinv()
        return (result.x, result.y, result.z)

    @property
    def Linv(self):
        return self.getLinv()

    def getVolume(self):
        """Return the box volume (area in 2D).

        Returns:
            float: Box volume.
        """
        return self.thisptr.getVolume()

    @property
    def volume(self):
        return self.getVolume()

    def getCoordinates(self, f):
        """Alias for :py:meth:`~.makeCoordinates()`

        .. deprecated:: 0.8
           Use :py:meth:`~.makeCoordinates()` instead.

        Args:
            f (:math:`\\left(3\\right)` :class:`numpy.ndarray`):
                Fractional coordinates :math:`\\left(x, y, z\\right)` between
                0 and 1 within parallelepipedal box.
        """
        return self.makeCoordinates(f)

    def makeCoordinates(self, f):
        """Convert fractional coordinates into real coordinates.

        Args:
            f (:math:`\\left(3\\right)` :class:`numpy.ndarray`):
                Fractional coordinates :math:`\\left(x, y, z\\right)` between
                0 and 1 within parallelepipedal box.

        Returns:
            list[float, float, float]: Vector of real coordinates
            :math:`\\left(x, y, z\\right)`.
        """
        cdef np.ndarray[float, ndim=1] l_vec = freud.common.convert_array(
            f, 1, dtype=np.float32, contiguous=True)
        cdef vec3[float] result = self.thisptr.makeCoordinates(
            <const vec3[float]> &l_vec[0])
        return [result.x, result.y, result.z]

    def makeFraction(self, vec):
        """Convert real coordinates into fractional coordinates.

        Args:
            vec (:math:`\\left(3\\right)` :class:`numpy.ndarray`):
                Real coordinates within parallelepipedal box.

        Returns:
            list[float, float, float]: A fractional coordinate vector.
        """
        cdef np.ndarray[float, ndim=1] l_vec = freud.common.convert_array(
            vec, 1, dtype=np.float32, contiguous=True)
        cdef vec3[float] result = self.thisptr.makeFraction(
            <const vec3[float]> &l_vec[0])
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
            <const vec3[float]> &l_vec[0])
        return [result.x, result.y, result.z]

    def getLatticeVector(self, i):
        """Get the lattice vector with index :math:`i`.

        Args:
            i (unsigned int):
                Index (:math:`0 \\leq i < d`) of the lattice vector, where
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
            vecs (:math:`\\left(3\\right)` or :math:`\\left(N, 3\\right)`
            :class:`numpy.ndarray`):
                Single vector or array of :math:`N` vectors. The vectors are
                altered in place and returned.

        Returns:
            :math:`\\left(3\\right)` or :math:`\\left(N, 3\\right)`
            :class:`numpy.ndarray`:
                Vectors wrapped into the box.
        """
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
        cdef vec3[float] result = self.thisptr.wrap(<vec3[float]> &l_vec[0])
        return (result.x, result.y, result.z)

    def unwrap(self, vecs, imgs):
        """Unwrap a given array of vectors inside the box back into real space,
        using an array of image indices that determine how many times to
        unwrap in each dimension.

        Args:
            vecs (:math:`\\left(3\\right)` or :math:`\\left(N, 3\\right)`
            :class:`numpy.ndarray`):
                Single vector or array of :math:`N` vectors. The vectors are
                modified in place.
            imgs (:math:`\\left(3\\right)` or :math:`\\left(N, 3\\right)`
            :class:`numpy.ndarray`):
                Single image index or array of :math:`N` image indices.

        Returns:
            :math:`\\left(3\\right)` or :math:`\\left(N, 3\\right)`
            :class:`numpy.ndarray`:
                Vectors unwrapped by the image indices provided.
        """
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
            <vec3[float]> &l_vec[0], <vec3[int]> &l_img[0])
        return [result.x, result.y, result.z]

    def getPeriodic(self):
        """Get the box's periodicity in each dimension.

        Returns:
            list[bool, bool, bool]: Periodic attributes in x, y, z.
        """
        periodic = self.thisptr.getPeriodic()
        return [periodic.x, periodic.y, periodic.z]

    def setPeriodic(self, x, y, z):
        """Set the box's periodicity in each dimension.

        Args:
            x (bool): True if periodic in x, False if not.
            y (bool): True if periodic in y, False if not.
            z (bool): True if periodic in z, False if not.
        """
        self.thisptr.setPeriodic(x, y, z)

    def getPeriodicX(self):
        """Get the box periodicity in the x direction.

        Returns:
            bool: True if periodic, False if not.
        """
        return self.thisptr.getPeriodicX()

    def setPeriodicX(self, val):
        """Set the box periodicity in the x direction.

        Args:
            val (bool): True if periodic, False if not.
        """
        return self.thisptr.setPeriodicX(val)

    def getPeriodicY(self):
        """Get the box periodicity in the y direction..

        Returns:
            bool: True if periodic, False if not.
        """
        return self.thisptr.getPeriodicY()

    def setPeriodicY(self, val):
        """Set the box periodicity in the y direction.

        Args:
            val (bool): True if periodic, False if not.
        """
        return self.thisptr.setPeriodicY(val)

    def getPeriodicZ(self):
        """Get the box periodicity in the z direction.

        Returns:
            bool: True if periodic, False if not.
        """
        return self.thisptr.getPeriodicZ()

    def setPeriodicZ(self, val):
        """Set the box periodicity in the z direction.

        Args:
            val (bool): True if periodic, False if not.
        """
        return self.thisptr.setPeriodicZ(val)

    def to_dict(self):
        """Return box as dictionary

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

    @classmethod
    def from_box(cls, box, dimensions=None):
        """Initialize a box instance from a box-like object.

        Args:
            box: A box-like object
            dimensions (int): Dimensionality of the box (Default value = None)

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
        if isinstance(box, np.ndarray) and box.shape == (3, 3):
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
            except (KeyError, TypeError):
                # Handle list-like
                Lx = box[0]
                Ly = box[1]
                Lz = box[2] if len(box) > 2 else 0
                xy, xz, yz = box[3:6] if len(box) >= 6 else (0, 0, 0)
        except Exception:
            raise ValueError(
                'Supplied box cannot be converted to type freud.box.Box')

        # The dimensions argument should override the box settings
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
            boxMatrix (array-like): A 3x3 matrix or list of lists
            dimensions (int):  Number of dimensions (Default value = None)
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
        a3x = np.dot(v0, v2) / Lx
        xz = a3x / Lz
        yz = (np.dot(v1, v2) - a2x * a3x) / (Ly * Lz)
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

    @property
    def L(self):
        return self.getL()

    @L.setter
    def L(self, value):
        self.setL(value)

    @property
    def Lx(self):
        return self.getLx()

    @Lx.setter
    def Lx(self, value):
        self.setL([value, self.Ly, self.Lz])

    @property
    def Ly(self):
        return self.getLy()

    @Ly.setter
    def Ly(self, value):
        self.setL([self.Lx, value, self.Lz])

    @property
    def Lz(self):
        return self.getLz()

    @Lz.setter
    def Lz(self, value):
        self.setL([self.Lx, self.Ly, value])

    @property
    def dimensions(self):
        return 2 if self.is2D() else 3

    @dimensions.setter
    def dimensions(self, value):
        assert value == 2 or value == 3
        self.set2D(value == 2)

    @property
    def periodic(self):
        return self.getPeriodic()

    @periodic.setter
    def periodic(self, periodic):
        self.setPeriodic(periodic[0], periodic[1], periodic[2])

    # Enable box to be pickled
    def __getinitargs__(self):
        return (self.getLx(), self.getLy(), self.getLz(),
                self.getTiltFactorXY(),
                self.getTiltFactorXZ(),
                self.getTiltFactorYZ(),
                self.is2D())

cdef BoxFromCPP(const box.Box & cppbox):
    return Box(cppbox.getLx(), cppbox.getLy(), cppbox.getLz(),
               cppbox.getTiltFactorXY(), cppbox.getTiltFactorXZ(),
               cppbox.getTiltFactorYZ(), cppbox.is2D())
