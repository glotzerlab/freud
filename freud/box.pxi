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
    imported by box.py for the user-facing API

    .. moduleauthor:: Richmond Newman <newmanrs@umich.edu>

    :param Lx: Length of side x
    :param Ly: Length of side y
    :param Lz: Length of side z
    :param xy: tilt of xy plane
    :param xz: tilt of xz plane
    :param yz: tilt of yz plane
    :param is2D: specify if box is 2D
    :type Lx: float
    :type Ly: float
    :type Lz: float
    :type xy: float
    :type xz: float
    :type yz: float
    :type is2D: bool
    """
    cdef box.Box * thisptr

    def __cinit__(self, Lx=None, Ly=None, Lz=None, xy=None, xz=None, yz=None,
                  is2D=None):
        # BEGIN Check for and warn about possible use of deprecated API
        # Should be removed in version version 0.7!
        args = (Lx, Ly, Lz, xy, xz, yz, is2D)
        if None in args:
            nargs = args.index(None)
            if nargs == 1:
                warnings.warn(
                    ("You may be using a deprecated Box constructor API!"
                        "Did you mean Box.cube()?"),
                    DeprecationWarning)
            elif nargs == 2 and isinstance(Ly, bool):
                raise ValueError(
                    ("You are using a deprecated Box constructor API!"
                        "Did you mean Box.square()?"))
            elif isinstance(Lz, bool) or isinstance(xy, bool) or isinstance(
                    xz, bool) or isinstance(yz, bool):
                raise ValueError(
                    "You are using a deprecated Box constructor API!")
        # END Check for and warn about possible use of deprecated API
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
                ("Specifying z-dimensions in a 2-dimensional box has"
                    "no effect!"))
        self.thisptr = new box.Box(Lx, Ly, Lz, xy, xz, yz, is2D)

    def __dealloc__(self):
        del self.thisptr

    def getL(self):
        """
        Return the lengths of the box as a tuple (x, y, z)

        :return: dimensions of the box as (x, y, z)
        :rtype: (float, float, float)
        """
        cdef vec3[float] result = self.thisptr.getL()
        return (result.x, result.y, result.z)

    def setL(self, L):
        """Set all side lengths of box to L

        :param L: Side length of box
        :type L: float
        """
        try:
            len(L)
        except TypeError:
            L = (L, L, L)

        if len(L) != 3:
            raise TypeError('Could not setL({})'.format(L))

        if self.is2D() and L[2] != 0:
            warnings.warn(
                ("Specifying z-dimensions in a 2-dimensional box has"
                    "no effect!"))
        self.thisptr.setL(L[0], L[1], L[2])

    def getLx(self):
        """Length of the x-dimension of the box

        :return: This box's x-dimension length
        :rtype: float
        """
        return self.thisptr.getLx()

    def getLy(self):
        """Length of the y-dimension of the box

        :return: This box's y-dimension length
        :rtype: float
        """
        return self.thisptr.getLy()

    def getLz(self):
        """Length of the z-dimension of the box

        :return: This box's z-dimension length
        :rtype: float
        """
        return self.thisptr.getLz()

    def getTiltFactorXY(self):
        """
        Return the tilt factor xy

        :return: xy tilt factor
        :rtype: float
        """
        return self.thisptr.getTiltFactorXY()

    @property
    def xy(self):
        """Tilt factor xy of the box

        :return: xy tilt factor
        :rtype: float
        """
        return self.getTiltFactorXY()

    def getTiltFactorXZ(self):
        """
        Return the tilt factor xz

        :return: xz tilt factor
        :rtype: float
        """
        return self.thisptr.getTiltFactorXZ()

    @property
    def xz(self):
        """Tilt factor xz of the box

        :return: xz tilt factor
        :rtype: float
        """
        return self.getTiltFactorXZ()

    def getTiltFactorYZ(self):
        """
        Return the tilt factor yz

        :return: yz tilt factor
        :rtype: float
        """
        return self.thisptr.getTiltFactorYZ()

    @property
    def yz(self):
        """Tilt factor yz of the box

        :return: yz tilt factor
        :rtype: float
        """
        return self.getTiltFactorYZ()

    def is2D(self):
        """Return if box is 2D (True) or 3D (False)

        :return: True if 2D, False if 3D
        :rtype: bool
        """
        return self.thisptr.is2D()

    def set2D(self, val):
        """
        Set the dimensionality to 2D (True) or 3D (False)

        :param val: 2D=True, 3D=False
        :type val: bool
        """
        self.thisptr.set2D(bool(val))

    @property
    def dimensions(self):
        """Number of dimensions of this box (only 2 or 3 are supported)

        :getter: Returns this box's number of dimensions
        :setter: Sets this box's number of dimensions
        :type: int
        """
        return 2 if self.is2D() else 3

    @dimensions.setter
    def dimensions(self, value):
        assert value == 2 or value == 3
        self.set2D(value == 2)

    def getLinv(self):
        """Return the inverse lengths of the box (1/Lx, 1/Ly, 1/Lz)

        :return: dimensions of the box as (1/Lx, 1/Ly, 1/Lz)
        :rtype: (float, float, float)
        """
        cdef vec3[float] result = self.thisptr.getLinv()
        return (result.x, result.y, result.z)

    @property
    def Linv(self):
        """Return the inverse lengths of the box (1/Lx, 1/Ly, 1/Lz)

        :return: dimensions of the box as (1/Lx, 1/Ly, 1/Lz)
        :rtype: (float, float, float)
        """
        return self.getLinv()

    def getVolume(self):
        """Return the box volume (area in 2D)

        :return: box volume
        :rtype: float
        """
        return self.thisptr.getVolume()

    @property
    def volume(self):
        """Return the box volume (area in 2D)

        :return: box volume
        :rtype: float
        """
        return self.getVolume()

    def getCoordinates(self, f):
        """
        Convert a vector of relative box coordinates (each in [0..1]) into
        absolute coordinates

        :param f: list[fx, fy, fz]
        :type f: list[float, float, float]
        :return: list[x, y, z]
        :rtype: list[float, float, float]
        """
        cdef vec3[float] fRaw = vec3[float](f[0], f[1], f[2])
        cdef vec3[float] resultVec = self.thisptr.makeCoordinates(
                < const vec3[float]&>fRaw)
        cdef float[3] result = [resultVec.x, resultVec.y, resultVec.z]
        return result

    def wrap(self, vecs):
        """
        Wrap a given array of vectors back into the box from python

        :param vecs: numpy array of vectors (Nx3) (or just 3 elements) to wrap
        :note: vecs returned in place (nothing returned)
        """
        if vecs.dtype != np.float32:
            raise ValueError("vecs must be a numpy float32 array")
        if len(vecs.shape) == 1:
            # only one vector to wrap
            vecs[:] = self._wrap(vecs)
        elif len(vecs.shape) == 2:
            # check to make sure the second dim is x, y, z
            if vecs.shape[1] != 3:
                raise ValueError(
                    "the 2nd dimension must have 3 values: x, y, z")
            for i, vec in enumerate(vecs):
                vecs[i] = self._wrap(vec)
        else:
            raise ValueError(
                ("Invalid dimensions given to box wrap. Wrap requires a 3"
                    "element array (3,), or (N,3) array as input"))

    def _wrap(self, vec):
        cdef np.ndarray[float, ndim = 1] l_vec = np.ascontiguousarray(
                vec.flatten())
        cdef vec3[float] result = self.thisptr.wrapMultiple(
                < vec3[float]&>l_vec[0])
        return (result.x, result.y, result.z)

    def unwrap(self, vecs, imgs):
        """
        Wrap a given array of vectors back into the box from python

        :param vecs: numpy array of vectors (Nx3) (or just 3 elements) to wrap
        :note: vecs returned in place (nothing returned)
        """
        if vecs.dtype != np.float32:
            raise ValueError("vecs must be a numpy float32 array")
        if imgs.dtype != np.int32:
            raise ValueError("imgs must be a numpy int32 array")
            # edit from here
        if len(vecs.shape) == 1:
            # only one vector to wrap
            # verify only one img
            if len(imgs.shape == 1):
                vecs = np.ascontiguousarray(
                    self._unwrap(vecs, imgs), dtype=np.float32)
            else:
                raise RuntimeError("imgs do not match vectors")
        elif len(vecs.shape) == 2:
            # check to make sure the second dim is x, y, z
            if vecs.shape[1] != 3:
                raise ValueError(
                    "the 2nd dimension must have 3 values: x, y, z")
            if len(imgs.shape) == 2:
                for i, (vec, img) in enumerate(zip(vecs, imgs)):
                    vecs[i] = self._unwrap(vec, img)
            else:
                raise RuntimeError("imgs do not match vectors")

    def _unwrap(self, vec, img):
        cdef np.ndarray[float, ndim = 1] l_vec = vec
        cdef np.ndarray[int, ndim = 1] l_img = img
        cdef vec3[float] result = self.thisptr.unwrap(
                < vec3[float]&>l_vec[0],
                < vec3[int]&>l_img[0])
        return [result.x, result.y, result.z]

    def makeCoordinates(self, f):
        """
        Convert fractional coordinates into real coordinates

        :param f: Fractional coordinates between 0 and 1 within
                    parallelpipedal box
        :type f: numpy.ndarray([x, y, z], dtype=numpy.float32)
        :return: A vector inside the box corresponding to f
        """
        cdef np.ndarray[float, ndim = 1] l_vec = np.ascontiguousarray(
                f.flatten())
        cdef vec3[float] result = self.thisptr.makeCoordinates(
                < const vec3[float]&>l_vec[0])
        return [result.x, result.y, result.z]

    def makeFraction(self, vec):
        """
        Convert fractional coordinates into real coordinates

        :param vec: Coordinates within parallelpipedal box
        :type vec: numpy.ndarray([x, y, z], dtype=numpy.float32)
        :return: Fractional vector inside the box corresponding to f
        """
        cdef np.ndarray[float, ndim = 1] l_vec = np.ascontiguousarray(
                vec.flatten())
        cdef vec3[float] result = self.thisptr.makeFraction(
                < const vec3[float]&>l_vec[0])
        return [result.x, result.y, result.z]

    def getLatticeVector(self, i):
        """
        Get the lattice vector with index i

        :param i: Index (0<=i<d) of the lattice vector, where d is dimension
                    (2 or 3)
        :type i: unsigned int
        :return: lattice vector with index i
        """
        cdef unsigned int index = i
        cdef vec3[float] result = self.thisptr.getLatticeVector(i)
        if self.thisptr.is2D():
            result.z = 0.0
        return [result.x, result.y, result.z]

    # Enable pickling of internal classes
    # Box
    def __getinitargs__(self):
        return (self.getLx(), self.getLy(), self.getLz(), self.is2D())

cdef BoxFromCPP(const box.Box & cppbox):
    """
    """
    return Box(cppbox.getLx(), cppbox.getLy(), cppbox.getLz(),
               cppbox.getTiltFactorXY(), cppbox.getTiltFactorXZ(),
               cppbox.getTiltFactorYZ(), cppbox.is2D())
