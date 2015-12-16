
from freud.util._Boost cimport shared_array
from freud.util._VectorMath cimport vec3, quat
from libcpp.complex cimport complex
import numpy as np
cimport numpy as np
cimport freud._kspace as kspace
from cython.operator cimport dereference

cdef class FTdelta:
    cdef kspace.FTdelta *thisptr
    # stored size of the fourier transform
    cdef unsigned int NK

    def __cinit__(self):
        self.thisptr = new kspace.FTdelta()
        self.NK = 0
        # keep copies of arrays we pass in since the C++ class just
        # keeps raw pointers
        self.K = self.r = self.q = None

    def __dealloc__(self):
        del self.thisptr

    def compute(self):
        self.thisptr.compute()

    def getFT(self):
        cdef (float complex)* ft_points = self.thisptr.getFT().get()
        if not self.NK:
            return np.array([[]], dtype=np.complex64)
        result = np.zeros([self.NK], dtype=np.complex64)
        cdef float complex[:] flatBuffer = <float complex[:self.NK]> ft_points
        result.flat[:] = flatBuffer
        return result

    def set_K(self, K):
        K = np.ascontiguousarray(K, dtype=np.float32)
        if K.ndim != 2 or K.shape[1] != 3:
            raise TypeError('K should be an Nx3 array')
        self.NK = K.shape[0]
        cdef np.ndarray[float] cK = K
        self.K = K
        self.thisptr.set_K(<vec3[float]*>cK.data, self.NK)

    def set_rq(self, r, q):
        r = np.ascontiguousarray(r, dtype=np.float32)
        if r.ndim != 2 or r.shape[1] != 3:
            raise TypeError('r should be an Nx3 array')
        q = np.ascontiguousarray(q, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != 4:
            raise TypeError('q should be an Nx4 array')
        if r.shape[0] != q.shape[0]:
            raise TypeError('r and q should have the same length')
        self.Np = r.shape[0]
        cdef np.ndarray[float] cr = r
        self.r = r
        cdef np.ndarray[float] cq = q
        self.q = q
        self.thisptr.set_rq(self.Np, <vec3[float]*>cr.data, <quat[float]*> cq.data)

    def set_density(self, float complex density):
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
        self.K = self.r = self.q = None

    def __dealloc__(self):
        del self.thisptr

    def compute(self):
        self.thisptr.compute()

    def getFT(self):
        cdef (float complex)* ft_points = self.thisptr.getFT().get()
        if not self.NK:
            return np.array([[]], dtype=np.complex64)
        result = np.zeros([self.NK], dtype=np.complex64)
        cdef float complex[:] flatBuffer = <float complex[:self.NK]> ft_points
        result.flat[:] = flatBuffer
        return result

    def set_K(self, K):
        K = np.ascontiguousarray(K, dtype=np.float32)
        if K.ndim != 2 or K.shape[1] != 3:
            raise TypeError('K should be an Nx3 array')
        self.NK = K.shape[0]
        cdef np.ndarray[float] cK = K
        self.K = K
        self.thisptr.set_K(<vec3[float]*>cK.data, self.NK)

    def set_rq(self, r, q):
        r = np.ascontiguousarray(r, dtype=np.float32)
        if r.ndim != 2 or r.shape[1] != 3:
            raise TypeError('r should be an Nx3 array')
        q = np.ascontiguousarray(q, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != 4:
            raise TypeError('q should be an Nx4 array')
        if r.shape[0] != q.shape[0]:
            raise TypeError('r and q should have the same length')
        self.Np = r.shape[0]
        cdef np.ndarray[float] cr = r
        self.r = r
        cdef np.ndarray[float] cq = q
        self.q = q
        self.thisptr.set_rq(self.Np, <vec3[float]*>cr.data, <quat[float]*> cq.data)

    def set_density(self, float complex density):
        self.thisptr.set_density(density)

    def set_radius(self, float radius):
        self.thisptr.set_radius(radius)
