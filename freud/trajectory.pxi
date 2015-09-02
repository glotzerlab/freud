
from freud.util._VectorMath cimport vec3
cimport freud._trajectory as trajectory

cdef class Box:
    """
    Freud box object. Wrapper for the c++ trajectory.Box() class

    :param L: Side length of Box
    :type L: float

    :param Lx: Length of side x
    :type Lx: float

    :param Ly: Length of side y
    :type Ly: float

    :param Lz: Length of side z
    :type Lz: float

    :param xy: tilt of xy plane
    :type xy: float

    :param xz: tilt of xz plane
    :type xz: float

    :param yz: tilt of yz plane
    :type yz: float

    :param is2D: specify if box is 2D
    :type is2D: bool
    """
    cdef trajectory.Box *thisptr

    def __cinit__(self, Lx=None, Ly=None, Lz=None, xy=None, xz=None, yz=None, is2D=None):

        # this is a work in progress
        # to support old API, allow for positional arguments, and determine how many there are
        argList = [Lx, Ly, Lz, xy, xz, yz, is2D]
        try:
            firstNone = argList.index(None)
            # if there are nones, need to determine if the rest are Nones
            # do not check is2D
            remainingArgs = argList[firstNone:-1]
            allNones = True
            for i in remainingArgs:
                if i is not None:
                    allNones = False
        except ValueError:
            # there are no nones
            allNones = False
        # old api
        if allNones == True:
            argList = argList[:firstNone]
            if len(argList) == 0:
                Lx = Ly = Lz = 0
                xy = xz = yz = 0
                is2D = False
            elif len(argList) == 1:
                Lx = Ly = Lz = argList[0]
                xy = xz = yz = 0
                is2D = False
            elif len(argList) == 2:
                Lx = Ly = Lz = argList[0]
                xy = xz = yz = 0
                is2D = argList[1]
            elif len(argList) == 3:
                # this assumes Lx, Ly, Lz; Lx, Ly, xy require kwargs
                Lx = argList[0]
                Ly = argList[1]
                Lz = argList[2]
                xy = xz = yz = 0
                is2D = False
            elif len(argList) == 4:
                # redundant for previous
                Lx = argList[0]
                Ly = argList[1]
                Lz = argList[2]
                xy = xz = yz = 0
                is2D = argList[3]
            elif len(argList) == 6:
                Lx = argList[0]
                Ly = argList[1]
                Lz = argList[2]
                xy = argList[3]
                xz = argList[4]
                yz = argList[5]
                is2D = False
            elif len(argList) == 7:
                Lx = argList[0]
                Ly = argList[1]
                Lz = argList[2]
                xy = argList[3]
                xz = argList[4]
                yz = argList[5]
                is2D = False
            else:
                raise TypeError('Could not create a Box with the given arguments: {}'.format(*argList))
        # new api
        else:
            # set is2D to bool
            if is2D is None:
                is2D = False
            # Lx must be spec'd
            if Lx is None:
                raise ValueError("Lx must be specified")
            if Ly is None:
                Ly = Lx
            if Lz is None:
                if is2D == True:
                    Lz = 0.0
                else:
                    Lz = Lx
            if xy is None:
                xy = 0.0
            if xz is None:
                xz = 0.0
            if yz is None:
                yz = 0.0
        # create the box
        self.thisptr = new trajectory.Box(Lx, Ly, Lz, xy, xz, yz, is2D)

    def __dealloc__(self):
        del self.thisptr

    def setL(self, L):
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

    ## Enable pickling of internal classes
    # Box
    def __getinitargs__(self):
        return (self.getLx(), self.getLy(), self.getLz(), self.is2D())

cdef BoxFromCPP(const trajectory.Box& cppbox):
    """
    """
    return Box(cppbox.getLx(), cppbox.getLy(), cppbox.getLz(), cppbox.getTiltFactorXY(), cppbox.getTiltFactorXZ(), cppbox.getTiltFactorYZ(), cppbox.is2D())
