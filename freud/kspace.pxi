
from freud.util._Boost cimport shared_array
from freud.util._VectorMath cimport vec3, quat
from libcpp.complex cimport complex
import numpy as np
cimport numpy as np
cimport freud._kspace as kspace
from cython.operator cimport dereference

cdef class FTdelta:
    """Compute the Fourier transform of a set of delta peaks at a list of
    K points."""
    cdef kspace.FTdelta *thisptr
    # stored size of the fourier transform
    cdef unsigned int NK

    def __cinit__(self):
        self.thisptr = new kspace.FTdelta()
        self.NK = 0
        # keep copies of arrays we pass in since the C++ class just
        # keeps raw pointers
        self.K = self.position = self.orientation = None

    def __dealloc__(self):
        del self.thisptr

    def compute(self):
        """Perform transform and store result internally"""
        self.thisptr.compute()

    def getFT(self):
        """Return the FT values"""
        cdef (float complex)* ft_points = self.thisptr.getFT().get()
        if not self.NK:
            return np.array([[]], dtype=np.complex64)
        result = np.zeros([self.NK], dtype=np.complex64)
        cdef float complex[:] flatBuffer = <float complex[:self.NK]> ft_points
        result.flat[:] = flatBuffer
        return result

    def set_K(self, K):
        """Set the K values to evaluate

        :param K: NKx3 array of K values to evaluate
        """
        K = np.ascontiguousarray(K, dtype=np.float32)
        if K.ndim != 2 or K.shape[1] != 3:
            raise TypeError('K should be an Nx3 array')
        self.NK = K.shape[0]
        cdef np.ndarray[float] cK = K
        self.K = K
        self.thisptr.set_K(<vec3[float]*>cK.data, self.NK)

    def set_rq(self, position, orientation):
        """Set particle positions and orientations

        :param position: Npx3 array of particle position vectors
        :param orientation: Npx4 array of particle orientation quaternions
        """
        position = np.ascontiguousarray(position, dtype=np.float32)
        if position.ndim != 2 or position.shape[1] != 3:
            raise TypeError('position should be an Nx3 array')
        orientation = np.ascontiguousarray(orientation, dtype=np.float32)
        if orientation.ndim != 2 or orientation.shape[1] != 4:
            raise TypeError('orientation should be an Nx4 array')
        if position.shape[0] != orientation.shape[0]:
            raise TypeError('position and orientation should have the same length')
        self.Np = position.shape[0]
        cdef np.ndarray[float] cr = position
        self.position = position
        cdef np.ndarray[float] cq = orientation
        self.orientation = orientation
        self.thisptr.set_rq(self.Np, <vec3[float]*>cr.data, <quat[float]*> cq.data)

    def set_density(self, float complex density):
        """Set scattering density

        :param density: complex value of scattering density
        """
        self.thisptr.set_density(density)

cdef class FTsphere:
    cdef kspace.FTsphere *thisptr
    # stored size of the fourier transform
    cdef unsigned int NK

    def __cinit__(self):
        self.thisptr = new kspace.FTsphere()
        self.NK = 0
        # keep copies of arrays we pass in since the C++ class just
        # keeps raw pointers
        self.K = self.position = self.orientation = None

    def __dealloc__(self):
        del self.thisptr

    def compute(self):
        """Perform transform and store result internally"""
        self.thisptr.compute()

    def getFT(self):
        """Return the FT values"""
        cdef (float complex)* ft_points = self.thisptr.getFT().get()
        if not self.NK:
            return np.array([[]], dtype=np.complex64)
        result = np.zeros([self.NK], dtype=np.complex64)
        cdef float complex[:] flatBuffer = <float complex[:self.NK]> ft_points
        result.flat[:] = flatBuffer
        return result

    def set_K(self, K):
        """Set the K values to evaluate

        :param K: NKx3 array of K values to evaluate
        """
        K = np.ascontiguousarray(K, dtype=np.float32)
        if K.ndim != 2 or K.shape[1] != 3:
            raise TypeError('K should be an Nx3 array')
        self.NK = K.shape[0]
        cdef np.ndarray[float] cK = K
        self.K = K
        self.thisptr.set_K(<vec3[float]*>cK.data, self.NK)

    def set_rq(self, position, orientation):
        """Set particle positions and orientations

        :param position: Npx3 array of particle position vectors
        :param orientation: Npx4 array of particle orientation quaternions
        """
        position = np.ascontiguousarray(position, dtype=np.float32)
        if position.ndim != 2 or position.shape[1] != 3:
            raise TypeError('position should be an Nx3 array')
        orientation = np.ascontiguousarray(orientation, dtype=np.float32)
        if orientation.ndim != 2 or orientation.shape[1] != 4:
            raise TypeError('orientation should be an Nx4 array')
        if position.shape[0] != orientation.shape[0]:
            raise TypeError('position and orientation should have the same length')
        self.Np = position.shape[0]
        cdef np.ndarray[float] cr = position
        self.position = position
        cdef np.ndarray[float] cq = orientation
        self.orientation = orientation
        self.thisptr.set_rq(self.Np, <vec3[float]*>cr.data, <quat[float]*> cq.data)

    def set_density(self, float complex density):
        """Set scattering density

        :param density: complex value of scattering density
        """
        self.thisptr.set_density(density)

    def set_radius(self, float radius):
        """Set particle volume according to radius"""
        self.thisptr.set_radius(radius)
