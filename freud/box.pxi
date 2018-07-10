# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

import warnings
import numpy as np
from freud.util._VectorMath cimport vec3
from libcpp.string cimport string
from libc.string cimport memcpy
cimport freud._box as box
cimport numpy as np

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class Box:
    """freud Box object. Wrapper for the C++ box.Box() class,
    imported by box.py for the user-facing API.
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
          (float, float, float): dimensions of the box as (x, y, z)

        """
        cdef vec3[float] result = self.thisptr.getL()
        return (result.x, result.y, result.z)

    def setL(self, L):
        """Set all side lengths of box to L.

        Args:
          L (float): Side length of box

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
          float: This box's x-dimension length

        """
        return self.thisptr.getLx()

    def getLy(self):
        """Length of the y-dimension of the box.

        Returns:
          float: This box's y-dimension length

        """
        return self.thisptr.getLy()

    def getLz(self):
        """Length of the z-dimension of the box.

        Returns:
          float: This box's z-dimension length

        """
        return self.thisptr.getLz()

    def getTiltFactorXY(self):
        """Return the tilt factor xy.

        Returns:
          float: xy tilt factor

        """
        return self.thisptr.getTiltFactorXY()

    @property
    def xy(self):
        """Tilt factor xy of the box.

        Returns:
          float: xy tilt factor

        """
        return self.getTiltFactorXY()

    def getTiltFactorXZ(self):
        """Return the tilt factor xz.

        Returns:
          float: xz tilt factor

        """
        return self.thisptr.getTiltFactorXZ()

    @property
    def xz(self):
        """Tilt factor xz of the box.

        Returns:
          float: xz tilt factor

        """
        return self.getTiltFactorXZ()

    def getTiltFactorYZ(self):
        """Return the tilt factor yz.

        Returns:
          float: yz tilt factor

        """
        return self.thisptr.getTiltFactorYZ()

    @property
    def yz(self):
        """Tilt factor yz of the box.

        Returns:
          float: yz tilt factor

        """
        return self.getTiltFactorYZ()

    def is2D(self):
        """Return if box is 2D (True) or 3D (False).

        Returns:
          bool: True if 2D, False if 3D

        """
        return self.thisptr.is2D()

    def set2D(self, val):
        """Set the dimensionality to 2D (True) or 3D (False).

        Args:
          val (bool): 2D=True, 3D=False

        """
        self.thisptr.set2D(bool(val))

    def getLinv(self):
        """Return the inverse lengths of the box (1/Lx, 1/Ly, 1/Lz).

        Returns:
          (float, float, float): dimensions of the box as (1/Lx, 1/Ly, 1/Lz)

        """
        cdef vec3[float] result = self.thisptr.getLinv()
        return (result.x, result.y, result.z)

    @property
    def Linv(self):
        """Return the inverse lengths of the box (1/Lx, 1/Ly, 1/Lz).

        Returns:
          (float, float, float): dimensions of the box as (1/Lx, 1/Ly, 1/Lz)

        """
        return self.getLinv()

    def getVolume(self):
        """Return the box volume (area in 2D).

        Returns:
          float: box volume

        """
        return self.thisptr.getVolume()

    @property
    def volume(self):
        """Return the box volume (area in 2D).

        Returns:
          float: box volume

        """
        return self.getVolume()

    def getCoordinates(self, f):
        """Alias for :py:meth:`~.makeCoordinates()`

        .. deprecated:: 0.8
           Use :py:meth:`~.makeCoordinates()` instead.

        Args:
          f (class:`numpy.ndarray`,
             shape= :math:`\\left(3\\right)`,
             dtype= :class:`numpy.float32`): Fractional coordinates
             :math:`\\left(x, y, z\\right)` between 0 and 1 within
             parallelepipedal box

        """
        return self.makeCoordinates(f)

    def makeCoordinates(self, f):
        """Convert fractional coordinates into real coordinates.

        Args:
          f (class:`numpy.ndarray`,
           shape= :math:`\\left(3\\right)`,
           dtype= :class:`numpy.float32`): Fractional coordinates :math:`\\left(x, y, z\\right)` between
           0 and 1 within parallelepipedal box

        Returns:
          list[float, float, float]: Vector of real coordinates :math:`\\left(x, y, z\\right)`

        """
        cdef np.ndarray[float, ndim=1] l_vec = freud.common.convert_array(
                f, 1, dtype=np.float32, contiguous=True)
        cdef vec3[float] result = self.thisptr.makeCoordinates(
                < const vec3[float]&>l_vec[0])
        return [result.x, result.y, result.z]

    def makeFraction(self, vec):
        """Convert real coordinates into fractional coordinates.

        Args:
          vec (class:`numpy.ndarray`,
               shape= :math:`\\left(3\\right)`,
               dtype= :class:`numpy.float32`): Real coordinates within parallelepipedal box

        Returns:
          A fractional coordinate vector

        """
        cdef np.ndarray[float, ndim=1] l_vec = freud.common.convert_array(
            vec, 1, dtype=np.float32, contiguous=True)
        cdef vec3[float] result = self.thisptr.makeFraction(
                < const vec3[float]&>l_vec[0])
        return [result.x, result.y, result.z]

    def getImage(self, vec):
        """Returns the image corresponding to a wrapped vector.

        .. versionadded:: 0.8

        Args:
          vec (class:`numpy.ndarray`,
               shape= :math:`\\left(3\\right)`,
               dtype= :class:`numpy.float32`): Coordinates of unwrapped vector

        Returns:
          class:`numpy.ndarray`,
          shape= :math:`\\left(3\\right)`,
          dtype= :class:`numpy.int32`: Image index vector

        """
        cdef np.ndarray[float, ndim=1] l_vec = freud.common.convert_array(
                vec, 1, dtype=np.float32, contiguous=True)
        cdef vec3[int] result = self.thisptr.getImage(
                < const vec3[float]&>l_vec[0])
        return [result.x, result.y, result.z]

    def getLatticeVector(self, i):
        """Get the lattice vector with index :math:`i`.

        Args:
          i (unsigned int): Index (:math:`0 \\leq i < d`) of the lattice vector, where
                            :math:`d` is the box dimension (2 or 3)

        Returns:
          lattice vector with index :math:`i`

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
          vecs (class:`numpy.ndarray`,
                shape= :math:`\\left(3\\right)` or
                :math:`\\left(N, 3\\right)`,
                dtype= :class:`numpy.float32`): Single vector or array of
                    :math:`N` vectors. Vecs are both altered in place and returned

        Returns:
          class:`numpy.ndarray`,
          shape= :math:`\\left(3\\right)` or
          :math:`\\left(N, 3\\right)`,
          dtype= :class:`numpy.float32`: vectors wrapped into the box

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
        """Wrap a single vector"""
        cdef np.ndarray[float, ndim=1] l_vec = vec
        cdef vec3[float] result = self.thisptr.wrap(
                < vec3[float]&>l_vec[0])
        return (result.x, result.y, result.z)

    def unwrap(self, vecs, imgs):
        """Unwrap a given array of vectors inside the box back into real space,
        using an array of image indices that determine how many times to
        unwrap in each dimension.

        Args:
          vecs (class:`numpy.ndarray`,
                shape= :math:`\\left(3\\right)` or
                :math:`\\left(N, 3\\right)`,
                dtype= :class:`numpy.float32`): Single vector or array of :math:`N` vectors. The vectors are modified in place.
          imgs (class:`numpy.ndarray`,
                shape= :math:`\\left(3\\right)` or
                :math:`\\left(N, 3\\right)`,
                dtype= :class:`numpy.float32`): Single image index or array of :math:`N` image indices

        Returns:
          class:`numpy.ndarray`,
          shape= :math:`\\left(3\\right)` or
          :math:`\\left(N, 3\\right)`,
          dtype= :class:`numpy.float32`: vectors unwrapped by the image indices provided

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
        """Unwrap a single vector"""
        cdef np.ndarray[float, ndim=1] l_vec = vec
        cdef np.ndarray[int, ndim=1] l_img = img
        cdef vec3[float] result = self.thisptr.unwrap(
                < vec3[float]&>l_vec[0],
                < vec3[int]&>l_img[0])
        return [result.x, result.y, result.z]

    def getPeriodic(self):
        """Get the box's periodicity in each dimension.

        Returns:
          list[bool, bool, bool]: Periodic attributes in x, y, z

        """
        periodic = self.thisptr.getPeriodic()
        return [periodic.x, periodic.y, periodic.z]

    def setPeriodic(self, x, y, z):
        """Set the box's periodicity in each dimension.

        Args:
          x (bool): True if periodic in x, False if not
          y (bool): True if periodic in y, False if not
          z (bool): True if periodic in z, False if not

        """
        self.thisptr.setPeriodic(x, y, z)

    def getPeriodicX(self):
        """Get the box periodicity in the x direction.

        Returns:
          bool: True if periodic, False if not

        """
        return self.thisptr.getPeriodicX()

    def setPeriodicX(self, val):
        """Set the box periodicity in the x direction.

        Args:
          val (bool): True if periodic, False if not

        """
        return self.thisptr.setPeriodicX(val)

    def getPeriodicY(self):
        """Get the box periodicity in the y direction.

        Returns:
          bool: True if periodic, False if not

        """
        return self.thisptr.getPeriodicY()

    def setPeriodicY(self, val):
        """Set the box periodicity in the y direction.

        Args:
          val (bool): True if periodic, False if not

        """
        return self.thisptr.setPeriodicY(val)

    def getPeriodicZ(self):
        """Get the box periodicity in the z direction.

        Returns:
          bool: True if periodic, False if not

        """
        return self.thisptr.getPeriodicZ()

    def setPeriodicZ(self, val):
        """Set the box periodicity in the z direction.

        Args:
          val (bool): True if periodic, False if not

        """
        return self.thisptr.setPeriodicZ(val)

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
