# distutils: language = c++
# cython: embedsignature=True

from freud.util._VectorMath cimport vec3
cimport freud.trajectory._trajectory as trajectory

cdef class Box:
    """
    Freud box object. Wrapper for the c++ trajectory.Box() class

    Constructs Box object from a variety of possible parameters:

    * L: If only L provided, constructs a cubic box with side length L
    * Lx, Ly, Lz: constructs an orthorhombic box with said side lengths
    * Lx, Ly, Lz, xy, xz, yz: constructs a triclinic box with side lengths and tilt factors
    * is2D: boolean to specify if 2D or 3D box

    """
    cdef trajectory.Box *thisptr

    def __cinit__(self, *args, is2D=None):
        if len(args) == 0:
            self.thisptr = new trajectory.Box()
        elif len(args) == 1:
            self.thisptr = new trajectory.Box(args[0], bool(is2D))
        elif len(args) == 3:
            self.thisptr = new trajectory.Box(args[0], args[1], args[2], is2D)
        elif len(args) == 4 and is2D is None:
            self.thisptr = new trajectory.Box(args[0], args[1], args[2], args[3])
        elif len(args) == 6:
            self.thisptr = new trajectory.Box(args[0], args[1], args[2], args[3], args[4], args[5], is2D)
        elif len(args) == 7 and is2D is None:
            self.thisptr = new trajectory.Box(args[0], args[1], args[2], args[3], args[4], args[5], args[6])
        else:
            raise TypeError('Could not create a Box with the given arguments: {}'.format(args))

    def __dealloc__(self):
        del self.thisptr

    def setL(self, L):
        """
        Sets the side length

        * L: side length
        """
        try:
            len(L)
        except TypeError:
            L = (L, L, L)

        if len(L) != 3:
            raise TypeError('Could not setL({})'.format(L))

        self.thisptr.setL(L[0], L[1], L[2])

    def set2D(self, val):
        self.thisptr.set2D(bool(val))

    def is2D(self):
        return self.thisptr.is2D()

    def getLx(self):
        return self.thisptr.getLx()

    def getLy(self):
        return self.thisptr.getLy()

    def getLz(self):
        return self.thisptr.getLz()

    def getL(self):
        cdef vec3[float] result = self.thisptr.getL()
        return (result.x, result.y, result.z)

    def getLinv(self):
        cdef vec3[float] result = self.thisptr.getLinv()
        return (result.x, result.y, result.z)

    def getTiltFactorXY(self):
        return self.thisptr.getTiltFactorXY()

    def getTiltFactorXZ(self):
        return self.thisptr.getTiltFactorXZ()

    def getTiltFactorYZ(self):
        return self.thisptr.getTiltFactorYZ()

    def getVolume(self):
        return self.thisptr.getVolume()

    def getCoordinates(self, f):
        """Convert a vector of relative box coordinates (each in [0..1]) into
        absolute coordinates"""
        cdef vec3[float] fRaw = vec3[float](f[0], f[1], f[2])
        cdef vec3[float] resultVec = self.thisptr.makeCoordinates(fRaw)
        cdef float[3] result = [resultVec.x, resultVec.y, resultVec.z]
        return result

cdef BoxFromCPP(const trajectory.Box& cppbox):
    """
    Function that returns a Python Box given a c++ Box
    """
    return Box(cppbox.getLx(), cppbox.getLy(), cppbox.getLz(), cppbox.getTiltFactorXY(), cppbox.getTiltFactorXZ(), cppbox.getTiltFactorYZ(), cppbox.is2D())
