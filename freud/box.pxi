# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

import warnings
from freud.util._VectorMath cimport vec3
cimport freud._box as box
import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libc.string cimport memcpy
# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class Box:
    """
    freud Box object. Wrapper for the c++ box.Box() class

    .. moduleauthor:: Richmond Newman <newmanrs@umich.edu>

    :param L: Side length of Box
    :param is2D: specify if box is 2D
    :param Lx: Length of side x
    :param Ly: Length of side y
    :param Lz: Length of side z
    :param xy: tilt of xy plane
    :param xz: tilt of xz plane
    :param yz: tilt of yz plane
    :param is2D: specify if box is 2D
    :type L: float
    :type is2D: bool
    :type Lx: float
    :type Ly: float
    :type Lz: float
    :type xy: float
    :type xz: float
    :type yz: float
    :type is2D: bool

    - Constructor calls:

        Initialize cubic box of side length L::

            freud.box.Box(L)

        Initialize cubic box of side length L (will create a 2D/3D box based on is2D)::

            freud.box.Box(L, is2D)

        Initialize orthorhombic box of side lengths Lx, Ly, Lz::

            freud.box.Box(Lx, Ly, Lz)

        Initializes box with side lengths Lx, Ly (, Lz if is2D=False)::

            freud.box.Box(Lx, Ly, is2D=False)

        Preferred method to initialize. Pass in as kwargs. Any not set will be set to the above defaults::

            freud.box.Box(Lx=0.0, Ly=0.0, Lz=0.0, xy=0.0, xz=0.0, yz=0.0, is2D=False)

    """
    cdef box.Box *thisptr

    def __cinit__(self, Lx=None, Ly=None, Lz=None, xy=None, xz=None, yz=None, is2D=None):
        # BEGIN Check for and warn about possible use of deprecated API
        # Should be removed in version version 0.7!
        args = (Lx, Ly, Lz, xy, xz, yz, is2D)
        if None in args:
            nargs = args.index(None)
            if nargs == 1:
                warnings.warn(
                    "You may be using a deprecated Box constructor API! Did you mean Box.cube()?",
                    DeprecationWarning)
            elif nargs == 2 and isinstance(Ly, bool):
                raise ValueError(
                  "You are using a deprecated Box constructor API! Did you mean Box.square()?")
            elif isinstance(Lz, bool) or isinstance(xy, bool) or isinstance(xz, bool) or isinstance(yz, bool):
                raise ValueError("You are using a deprecated Box constructor API!")
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
        if is2D and (Lz != 0 or xz != 0 or yz!= 0):
            warnings.warn("Specifying z-dimensions in a 2-dimensional box has no effect!")
        self.thisptr = new box.Box(Lx, Ly, Lz, xy, xz, yz, is2D)

    def __dealloc__(self):
        del self.thisptr

    def setL(self, L):
        """
        Set all side lengths of box to L

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
            warnings.warn("Specifying z-dimensions in a 2-dimensional box has no effect!")
        self.thisptr.setL(L[0], L[1], L[2])

    def set2D(self, val):
        """
        Set the dimensionality to 2D (True) or 3D (False)

        :param val: 2D=True, 3D=False
        :type val: bool
        """
        self.thisptr.set2D(bool(val))

    def is2D(self):
        """
        return if box is 2D (True) or 3D (False)

        :return: True if 2D, False if 3D
        :rtype: bool
        """
        return self.thisptr.is2D()

    @property
    def Lx(self):
        """Return the length of the x-dimension of the box
        """
        return self.getLx()

    def getLx(self):
        """
        return the length of the x-dimension of the box

        :return: x-dimension of the box
        :rtype: float
        """
        return self.thisptr.getLx()

    @property
    def Ly(self):
        """Return the length of the y-dimension of the box
        """
        return self.getLy()

    def getLy(self):
        """
        return the length of the y-dimension of the box

        :return: y-dimension of the box
        :rtype: float
        """
        return self.thisptr.getLy()

    @property
    def Lz(self):
        """Return the length of the z-dimension of the box
        """
        return self.getLz()

    def getLz(self):
        """
        return the length of the z-dimension of the box

        :return: z-dimension of the box
        :rtype: float
        """
        return self.thisptr.getLz()

    @property
    def L(self):
        """Return the lengths of the box as a tuple (x, y, z)
        """
        return self.getL()

    @L.setter
    def L(self, value):
        """Set all side lengths of box to L
        """
        self.setL(value)

    def getL(self):
        """
        return the lengths of the box as a tuple (x, y, z)

        :return: dimensions of the box as (x, y, z)
        :rtype: (float, float, float)
        """
        cdef vec3[float] result = self.thisptr.getL()
        return (result.x, result.y, result.z)

    @property
    def Linv(self):
        """Return the inverse lengths of the box (1/x, 1/y, 1/z)
        """
        return self.getLinv()

    def getLinv(self):
        """
        return the inverse lengths of the box (1/x, 1/y, 1/z)

        :return: dimensions of the box as (1/x, 1/y, 1/z)
        :rtype: (float, float, float)
        """
        cdef vec3[float] result = self.thisptr.getLinv()
        return (result.x, result.y, result.z)

    @property
    def tilt_factor_xy(self):
        """Return the tilt factor xy
        """
        return self.getTiltFactorXY()

    def getTiltFactorXY(self):
        """
        return the tilt factor xy

        :return: xy tilt factor
        :rtype: float
        """
        return self.thisptr.getTiltFactorXY()

    @property
    def tilt_factor_xz(self):
        """Return the tilt factor xz
        """
        return self.getTiltFactorXZ()

    def getTiltFactorXZ(self):
        """
        return the tilt factor xz

        :return: xz tilt factor
        :rtype: float
        """
        return self.thisptr.getTiltFactorXZ()

    @property
    def tilt_factor_yz(self):
        """Return the tilt factor yz
        """
        return self.getTiltFactorYZ()

    def getTiltFactorYZ(self):
        """
        return the tilt factor yz

        :return: yz tilt factor
        :rtype: float
        """
        return self.thisptr.getTiltFactorYZ()

    @property
    def volume(self):
        """Return the box volume
        """
        return self.getVolume()

    def getVolume(self):
        """
        return the box volume

        :return: box volume
        :rtype: float
        """
        return self.thisptr.getVolume()

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
        cdef vec3[float] resultVec = self.thisptr.makeCoordinates(<const vec3[float]&>fRaw)
        # check on this
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
                raise ValueError("the 2nd dimension must have 3 values: x, y, z")
            for i, vec in enumerate(vecs):
                vecs[i] = self._wrap(vec)
        else:
            raise ValueError("Invalid dimensions given to box wrap. Wrap requires a 3 element array (3,), or (N,3) array as input")

    def _wrap(self, vec):
        cdef np.ndarray[float,ndim=1] l_vec = np.ascontiguousarray(vec.flatten())
        cdef vec3[float] result = self.thisptr.wrapMultiple(<vec3[float]&>l_vec[0])
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
                vecs = np.ascontiguousarray(self._unwrap(vecs, imgs), dtype=np.float32)
            else:
                raise RuntimeError("imgs do not match vectors")
        elif len(vecs.shape) == 2:
            # check to make sure the second dim is x, y, z
            if vecs.shape[1] != 3:
                raise ValueError("the 2nd dimension must have 3 values: x, y, z")
            if len(imgs.shape) == 2:
                for i, (vec, img) in enumerate(zip(vecs, imgs)):
                    vecs[i] = self._unwrap(vec, img)
            else:
                raise RuntimeError("imgs do not match vectors")

    def _unwrap(self, vec, img):
        cdef np.ndarray[float,ndim=1] l_vec = vec
        cdef np.ndarray[int,ndim=1] l_img = img
        cdef vec3[float] result = self.thisptr.unwrap(<vec3[float]&>l_vec[0], <vec3[int]&>l_img[0])
        return [result.x, result.y, result.z]

    def makeCoordinates(self, f):
        """
        Convert fractional coordinates into real coordinates

        :param f: Fractional coordinates between 0 and 1 within parallelpipedal box
        :type f: numpy.ndarray([x, y, z], dtype=numpy.float32)
        :return: A vector inside the box corresponding to f
        """
        cdef np.ndarray[float,ndim=1] l_vec = np.ascontiguousarray(f.flatten())
        cdef vec3[float] result = self.thisptr.makeCoordinates(<const vec3[float]&>l_vec[0])
        return [result.x, result.y, result.z]

    def makeFraction(self, vec):
        """
        Convert fractional coordinates into real coordinates

        :param vec: Coordinates within parallelpipedal box
        :type vec: numpy.ndarray([x, y, z], dtype=numpy.float32)
        :return: Fractional vector inside the box corresponding to f
        """
        cdef np.ndarray[float,ndim=1] l_vec = np.ascontiguousarray(vec.flatten())
        cdef vec3[float] result = self.thisptr.makeFraction(<const vec3[float]&>l_vec[0])
        return [result.x, result.y, result.z]

    def getLatticeVector(self, i):
        """
        Get the lattice vector with index i

        :param i: Index (0<=i<d) of the lattice vector, where d is dimension (2 or 3)
        :type i: unsigned int
        :return: lattice vector with index i
        """
        cdef unsigned int index = i
        cdef vec3[float] result = self.thisptr.getLatticeVector(i)
        if self.thisptr.is2D():
            result.z = 0.0
        return [result.x, result.y, result.z]

    ## Enable pickling of internal classes
    # Box
    def __getinitargs__(self):
        return (self.getLx(), self.getLy(), self.getLz(), self.is2D())

cdef BoxFromCPP(const box.Box& cppbox):
    """
    """
    return Box(cppbox.getLx(), cppbox.getLy(), cppbox.getLz(), cppbox.getTiltFactorXY(), cppbox.getTiltFactorXZ(), cppbox.getTiltFactorYZ(), cppbox.is2D())
