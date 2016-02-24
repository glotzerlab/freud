
from freud.util._VectorMath cimport vec3
cimport freud._trajectory as trajectory
import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libc.string cimport memcpy
# Numpy must be initialized. When using numpy from C or Cython you must 
# _always_ do that, or you will have segfaults 
np.import_array()

cdef class Box:
    """
    Freud box object. Wrapper for the c++ trajectory.Box() class

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

            freud.trajectory.Box(L)

        Initialize cubic box of side length L (will create a 2D/3D box based on is2D)::

            freud.trajectory.Box(L, is2D)

        Initialize orthorhombic box of side lengths Lx, Ly, Lz::

            freud.trajectory.Box(Lx, Ly, Lz)

        Initializes box with side lengths Lx, Ly (, Lz if is2D=False)::

            freud.trajectory.Box(Lx, Ly, is2D=False)

        Preferred method to initialize. Pass in as kwargs. Any not set will be set to the above defaults::

            freud.trajectory.Box(Lx=0.0, Ly=0.0, Lz=0.0, xy=0.0, xz=0.0, yz=0.0, is2D=False)

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

    def getLx(self):
        """
        return the length of the x-dimension of the box

        :return: x-dimension of the box
        :rtype: float
        """
        return self.thisptr.getLx()

    def getLy(self):
        """
        return the length of the y-dimension of the box

        :return: y-dimension of the box
        :rtype: float
        """
        return self.thisptr.getLy()

    def getLz(self):
        """
        return the length of the z-dimension of the box

        :return: z-dimension of the box
        :rtype: float
        """
        return self.thisptr.getLz()

    def getL(self):
        """
        return the lengths of the box as a tuple (x, y, z)

        :return: dimensions of the box as (x, y, z)
        :rtype: (float, float, float)
        """
        cdef vec3[float] result = self.thisptr.getL()
        return (result.x, result.y, result.z)

    def getLinv(self):
        """
        return the inverse lengths of the box (1/x, 1/y, 1/z)

        :return: dimensions of the box as (1/x, 1/y, 1/z)
        :rtype: (float, float, float)
        """
        cdef vec3[float] result = self.thisptr.getLinv()
        return (result.x, result.y, result.z)

    def getTiltFactorXY(self):
        """
        return the tilt factor xy

        :return: xy tilt factor
        :rtype: float
        """
        return self.thisptr.getTiltFactorXY()

    def getTiltFactorXZ(self):
        """
        return the tilt factor xz

        :return: xz tilt factor
        :rtype: float
        """
        return self.thisptr.getTiltFactorXZ()

    def getTiltFactorYZ(self):
        """
        return the tilt factor yz

        :return: yz tilt factor
        :rtype: float
        """
        return self.thisptr.getTiltFactorYZ()

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
            vecs = np.ascontiguousarray(self._wrap(vecs), dtype=np.float32)
        elif len(vecs.shape) == 2:
            # check to make sure the second dim is x, y, z
            if vecs.shape[1] != 3:
                raise ValueError("the 2nd dimension must have 3 values: x, y, z")
            for i, vec in enumerate(vecs):
                vecs[i] = self._wrap(vec)

    def _wrap(self, vec):
        cdef np.ndarray l_vec = vec
        cdef vec3[float] result = self.thisptr.wrap(<vec3[float]&>l_vec.data)
        return [result.x, result.y, result.z]

    def makeCoordinates(self, f):
        """
        Convert fractional coordinates into real coordinates

        :param f: Fractional coordinates between 0 and 1 within parallelpipedal box
        :type f: numpy.ndarray([x, y, z], dtype=numpy.float32)
        :return: A vector inside the box corresponding to f
        """
        cdef np.ndarray l_vec = f
        cdef vec3[float] result = self.thisptr.makeCoordinates(<const vec3[float]&>l_vec.data)
        return [result.x, result.y, result.z]

    def makeFraction(self, vec):
        """
        Convert fractional coordinates into real coordinates

        :param vec: Coordinates within parallelpipedal box
        :type vec: numpy.ndarray([x, y, z], dtype=numpy.float32)
        :return: Fractional vector inside the box corresponding to f
        """
        cdef np.ndarray l_vec = vec
        cdef vec3[float] result = self.thisptr.makeFraction(<const vec3[float]&>l_vec.data)
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

cdef BoxFromCPP(const trajectory.Box& cppbox):
    """
    """
    return Box(cppbox.getLx(), cppbox.getLy(), cppbox.getLz(), cppbox.getTiltFactorXY(), cppbox.getTiltFactorXZ(), cppbox.getTiltFactorYZ(), cppbox.is2D())

cdef class DCDLoader:
    """
    Freud DCDLoader. Wrapper for the c++ trajectory.DCDLoader() class

    :param dcd_fname: name of dcd file
    :type dcd_fname: string

    :note: It is note expected that users should need to load dcd files outside of the trajectory readers

    """
    cdef trajectory.DCDLoader *thisptr

    def __cinit__(self, dcd_fname):

        cdef string l_dcd_fname = dcd_fname.encode('UTF-8')
        self.thisptr = new trajectory.DCDLoader(<const string&>l_dcd_fname)

    def __dealloc__(self):
        del self.thisptr

    def jumpToFrame(self, frame_idx):
        """
        :param frame: Frame number to jump to
        :note: The molfile plugins only support skipping forward in the file. As such, \
        :py:meth:`freud.trajectory.jumpToFrame(frame)` must reload the file from scratch if given a previous frame \
        number than the current
        """
        if not type(frame_idx) == int:
            raise TypeError("frame_idx must be an integer")
        self.thisptr.jumpToFrame(frame_idx)

    def readNextFrame(self):
        """
        Reads the next frame from the DCD file
        """
        self.thisptr.readNextFrame()

    def getBox(self):
        """
        :return: Freud Box
        :rtype: :py:meth:`freud.trajectory.Box()`
        """
        return BoxFromCPP(<trajectory.Box> self.thisptr.getBox())

    def getNumParticles(self):
        """
        Get the number of particles in the dcd file.

        :return: number of particles
        :rtype: int
        """
        return self.thisptr.getNumParticles()

    def getLastFrameNum(self):
        """
        Get the last frame read.

        :return: index of the last read frame
        :rtype: int
        """
        return self.thisptr.getLastFrameNum()

    def getFrameCount(self):
        """
        Get the number of frames.

        :return: number of frames in dcd file
        :rtype: int
        """
        return self.thisptr.getFrameCount()

    def getFileName(self):
        """
        Get the name of the dcd file.

        :return: name of the dcd file
        :rtype: str
        """
        cdef string fname = self.thisptr.getFileName()
        return fname.decode('UTF-8')

    def getTimeStep(self):
        """
        Get the timestep.

        :return: timestep
        :rtype: int
        :todo: which timestep is returned?
        """
        return self.thisptr.getTimeStep()

    def getPoints(self):
        """
        Access the points read by the last step.

        :return: points from the previous timestep
        :rtype: np.ndarray(shape=[N, 3], dtype=np.float32)
        """
        cdef float *points = self.thisptr.getPoints().get()
        # this needs to be a copy, otherwise it may pass out of scope and result in a seg fault
        cdef np.ndarray[np.float32_t,ndim=2] result = np.zeros(shape=(self.thisptr.getNumParticles(),3),dtype=np.float32)
        memcpy(result.data,points,result.nbytes)
        return result
