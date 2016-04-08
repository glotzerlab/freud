
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
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.NK
        result = np.PyArray_SimpleNewFromData(1, shape, np.NPY_COMPLEX64, ft_points)
        return result

    def set_K(self, K):
        """Set the K values to evaluate

        :param K: NKx3 array of K values to evaluate
        """
        K = np.ascontiguousarray(K, dtype=np.float32)
        if K.ndim != 2 or K.shape[1] != 3:
            raise TypeError('K should be an Nx3 array')
        self.NK = K.shape[0]
        cdef np.ndarray[float,ndim=2] cK = K
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
        Np = position.shape[0]
        cdef np.ndarray[float,ndim=2] cr = position
        cdef np.ndarray[float,ndim=2] cq = orientation
        self.thisptr.set_rq(Np, <vec3[float]*>cr.data, <quat[float]*> cq.data)

    def set_density(self, float complex density):
        """Set scattering density

        :param density: complex value of scattering density
        """
        self.thisptr.set_density(density)

    def set_radius(self, float radius):
        """Set particle volume according to radius"""
        self.thisptr.set_radius(radius)

cdef class FTpolyhedron:
    cdef kspace.FTpolyhedron *thisptr
    # stored size of the fourier transform
    cdef unsigned int NK

    def __cinit__(self):
        self.thisptr = new kspace.FTpolyhedron()
        self.NK = 0

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
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.NK
        result = np.PyArray_SimpleNewFromData(1, shape, np.NPY_COMPLEX64, ft_points)
        return result

    def set_K(self, K):
        """Set the K values to evaluate

        :param K: NKx3 array of K values to evaluate
        """
        K = np.ascontiguousarray(K, dtype=np.float32)
        if K.ndim != 2 or K.shape[1] != 3:
            raise TypeError('K should be an Nx3 array')
        self.NK = K.shape[0]
        cdef np.ndarray[float,ndim=2] cK = K
        self.thisptr.set_K(<vec3[float]*>cK.data, self.NK)

    def set_params(self, verts, facet_offs, facets, norms,d, area, volume):
        """Set polyhedron geometry

        :param verts: Nvx3 array of vertex coordinates
        :param facet_offs: (Nf+1)x1 array of facet start offsets
        :param facets: Nfvx1 array of facet vertex indices
        :param norms: Nfx3 array of facet normals
        :param d: Nfx1 array of facet distances
        :param area: Nfx1 array of facet areas
        :param volume: polyhedron volume
        """

        verts = np.ascontiguousarray(verts, dtype=np.float32)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise TypeError('position should be an Nx3 array')

        facet_offs = np.ascontiguousarray(facet_offs, dtype=np.uint32)
        if facet_offs.ndim != 1:
            raise TypeError('facet_offs should be Nx1 array')
        facets = np.ascontiguousarray(facets, dtype=np.uint32)
        if facets.ndim != 1:
            raise TypeError('facets should be Nx1 array')
        norms = np.ascontiguousarray(norms, dtype=np.float32)
        if norms.ndim != 2 and norms.shape[1] != 3:
            raise TypeError('norms should be Nx3 array')
        d = np.ascontiguousarray(d, dtype=np.float32)
        if d.ndim != 1:
            raise TypeError('d should be Nx1 array')
        area = np.ascontiguousarray(area, dtype=np.float32)
        if area.ndim != 1:
            raise TypeError('area should be Nx1 array')
        if norms.shape[0] != facet_offs.shape[0] - 1:
            raise RuntimeError('Length of norms should be equal to number of facet offsets - 1')
        if d.shape[0] != facet_offs.shape[0] - 1:
            raise RuntimeError('Length of facet distances should be equal to number of facet offsets - 1')
        if area.shape[0] != facet_offs.shape[0] - 1:
            raise RuntimeError('Length of areas should be equal to number of facet offsets - 1')
        volume = float(volume)
        cdef np.ndarray[float,ndim=2] cverts = verts
        cdef np.ndarray[unsigned int] cfacet_offs = facet_offs
        cdef np.ndarray[unsigned int] cfacets = facets
        cdef np.ndarray[float,ndim=2] cnorms = norms
        cdef np.ndarray[float] cd = d
        cdef np.ndarray[float] carea = area
        self.thisptr.set_params(verts.shape[0], <vec3[float]*>cverts.data, facet_offs.shape[0] - 1, \
            <unsigned int*>cfacet_offs.data, <unsigned int*>cfacets.data, <vec3[float]*>cnorms.data, \
            <float*>cd.data, <float*>carea.data,volume)

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
        Np = position.shape[0]
        cdef np.ndarray[float,ndim=2] cr = position
        cdef np.ndarray[float,ndim=2] cq = orientation
        self.thisptr.set_rq(Np, <vec3[float]*>cr.data, <quat[float]*> cq.data)

    def set_density(self, float complex density):
        """Set scattering density

        :param density: complex value of scattering density
        """
        self.thisptr.set_density(density)
