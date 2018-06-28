# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

import numpy as np
from freud.util._VectorMath cimport vec3, quat
from libcpp.complex cimport complex
from cython.operator cimport dereference
cimport numpy as np
cimport freud._kspace as kspace

cdef class FTdelta:
    """Compute the Fourier transform of a set of delta peaks at a list of
    :math:`K` points.

    .. moduleauthor:: Jens Glaser <jsglaser@umich.edu>

    """
    cdef kspace.FTdelta * thisptr
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
        return self

    @property
    def FT(self):
        """Return the FT values"""
        return self.getFT()

    def getFT(self):
        """Return the FT values"""
        cdef(float complex) * ft_points = self.thisptr.getFT().get()
        if not self.NK:
            return np.array([[]], dtype=np.complex64)
        result = np.zeros([self.NK], dtype=np.complex64)
        cdef float complex[:] flatBuffer = <float complex[:self.NK] > ft_points
        result.flat[:] = flatBuffer
        return result

    def set_K(self, K):
        """Set the :math:`K` values to evaluate

        :param K: :math:`K` values to evaluate
        :type K: :class:`numpy.ndarray`,
                 shape=(:math:`N_{K}`, 3),
                 dtype= :class:`numpy.float32`
        """
        K = freud.common.convert_array(
                K, 1, dtype=np.float32, contiguous=True,
                array_name="K")
        if K.shape[1] != 3:
            raise TypeError('K should be an Nx3 array')

        self.NK = K.shape[0]
        # cdef unsigned int NK = <unsigned int> K.shape[0]
        cdef np.ndarray[float, ndim= 2] cK = K
        self.thisptr.set_K(< vec3[float]*>cK.data, self.NK)

    def set_rq(self, position, orientation):
        """Set particle positions and orientations.

        :param position: particle position vectors
        :param orientation: particle orientation quaternions
        :type position: :class:`numpy.ndarray`,
                        shape=(:math:`N_{particles}`, 3),
                        dtype= :class:`numpy.float32`
        :type orientation: :class:`numpy.ndarray`,
                           shape=(:math:`N_{particles}`, 4),
                           dtype= :class:`numpy.float32`
        """
        position = freud.common.convert_array(
                position, 2, dtype=np.float32, contiguous=True,
                array_name="position")
        orientation = freud.common.convert_array(
                orientation, 2, dtype=np.float32, contiguous=True,
                array_name="orientation")
        if position.shape[1] != 3:
            raise TypeError('position should be an Nx3 array')
        if orientation.shape[1] != 4:
            raise TypeError('orientation should be an Nx4 array')
        if position.shape[0] != orientation.shape[0]:
            raise TypeError(
                'position and orientation should have the same length')
        cdef unsigned int Np = <unsigned int > position.shape[0]
        cdef np.ndarray[float, ndim= 2] cr = position
        cdef np.ndarray[float, ndim= 2] cq = orientation
        self.thisptr.set_rq(Np, < vec3[float]*>cr.data,
                            < quat[float]*> cq.data)

    def set_density(self, float complex density):
        """Set scattering density.

        :param density: complex value of scattering density
        """
        self.thisptr.set_density(density)

cdef class FTsphere:
    """
    .. moduleauthor:: Jens Glaser <jsglaser@umich.edu>
    """
    cdef kspace.FTsphere * thisptr
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
        return self

    @property
    def FT(self):
        """Return the FT values"""
        return self.getFT()

    def getFT(self):
        """Return the FT values"""
        cdef(float complex) * ft_points = self.thisptr.getFT().get()
        if not self.NK:
            return np.array([[]], dtype=np.complex64)
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp > self.NK
        result = np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_COMPLEX64, ft_points)
        return result

    def set_K(self, K):
        """Set the :math:`K` values to evaluate

        :param K: :math:`K` values to evaluate
        :type K: :class:`numpy.ndarray`,
                    shape=(:math:`N_{K}`, 3),
                    dtype= :class:`numpy.float32`
        """
        K = np.ascontiguousarray(K, dtype=np.float32)
        if K.ndim != 2 or K.shape[1] != 3:
            raise TypeError('K should be an Nx3 array')
        self.NK = K.shape[0]
        cdef np.ndarray[float, ndim = 2] cK = K
        self.thisptr.set_K(< vec3[float]*>cK.data, self.NK)

    def set_rq(self, position, orientation):
        """Set particle positions and orientations.

        :param position: particle position vectors
        :param orientation: particle orientation quaternions
        :type position: :class:`numpy.ndarray`,
                        shape=(:math:`N_{particles}`, 3),
                        dtype= :class:`numpy.float32`
        :type orientation: :class:`numpy.ndarray`,
                            shape=(:math:`N_{particles}`, 4),
                            dtype= :class:`numpy.float32`
        """
        position = freud.common.convert_array(
                position, 2, dtype=np.float32, contiguous=True,
                array_name="position")
        orientation = freud.common.convert_array(
                orientation, 2, dtype=np.float32, contiguous=True,
                array_name="orientation")
        if position.shape[1] != 3:
            raise TypeError('position should be an Nx3 array')
        if orientation.shape[1] != 4:
            raise TypeError('orientation should be an Nx4 array')
        if position.shape[0] != orientation.shape[0]:
            raise TypeError(
                'position and orientation should have the same length')
        Np = position.shape[0]
        cdef np.ndarray[float, ndim = 2] cr = position
        cdef np.ndarray[float, ndim = 2] cq = orientation
        self.thisptr.set_rq(Np, < vec3[float]*>cr.data,
                            < quat[float]*> cq.data)

    def set_density(self, float complex density):
        """Set scattering density.

        :param density: complex value of scattering density
        :type density: float complex
        """
        self.thisptr.set_density(density)

    def set_radius(self, float radius):
        """Set particle volume according to radius"""
        self.thisptr.set_radius(radius)

cdef class FTpolyhedron:
    """
    .. moduleauthor:: Jens Glaser <jsglaser@umich.edu>
    """
    cdef kspace.FTpolyhedron * thisptr
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
        return self

    @property
    def FT(self):
        """Return the FT values"""
        return self.getFT()

    def getFT(self):
        """Return the FT values"""
        cdef(float complex) * ft_points = self.thisptr.getFT().get()
        if not self.NK:
            return np.array([[]], dtype=np.complex64)
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp > self.NK
        result = np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_COMPLEX64, ft_points)
        return result

    def set_K(self, K):
        """Set the :math:`K` values to evaluate

        :param K: :math:`K` values to evaluate
        :type K: :class:`numpy.ndarray`,
                    shape=(:math:`N_{K}`, 3),
                    dtype= :class:`numpy.float32`
        """
        K = freud.common.convert_array(
            K, 2, dtype=np.float32, contiguous=True,
            array_name="K")
        if K.shape[1] != 3:
            raise TypeError('K should be an Nx3 array')
        self.NK = K.shape[0]
        cdef np.ndarray[float, ndim = 2] cK = K
        self.thisptr.set_K(< vec3[float]*>cK.data, self.NK)

    def set_params(self, verts, facet_offs, facets, norms, d, area, volume):
        """Set polyhedron geometry.

        :param verts: vertex coordinates
        :param facet_offs: facet start offsets
        :param facets: facet vertex indices
        :param norms: facet normals
        :param d: facet distances
        :param area: facet areas
        :param volume: polyhedron volume
        :type verts: :class:`numpy.ndarray`,
                        shape=(:math:`N_{vertices}`, 3),
                        dtype= :class:`numpy.float32`
        :type facet_offs: :class:`numpy.ndarray`,
                            shape=(:math:`N_{facets}`),
                            dtype= :class:`numpy.float32`
        :type facets: :class:`numpy.ndarray`,
                        shape=(:math:`N_{facets}`),
                        dtype= :class:`numpy.float32`
        :type norms: :class:`numpy.ndarray`,
                        shape=(:math:`N_{facets}`, 3),
                        dtype= :class:`numpy.float32`
        :type d: :class:`numpy.ndarray`,
                    shape=(:math:`N_{facets}`),
                    dtype= :class:`numpy.float32`
        :type area: :class:`numpy.ndarray`,
                    shape=(:math:`N_{facets}`),
                    dtype= :class:`numpy.float32`
        :type volume: float
        """
        verts = freud.common.convert_array(
                verts, 2, dtype=np.float32, contiguous=True,
                array_name="verts")
        if verts.shape[1] != 3:
            raise TypeError('verts should be an Nx3 array')

        facet_offs = freud.common.convert_array(
                facet_offs, 1, dtype=np.uint32, contiguous=True,
                array_name="facet_offs")

        facets = freud.common.convert_array(
                facets, 1, dtype=np.uint32, contiguous=True,
                array_name="facets")

        norms = freud.common.convert_array(
                norms, 2, dtype=np.float32, contiguous=True,
                array_name="norms")
        if norms.shape[1] != 3:
            raise TypeError('norms should be an Nx3 array')

        d = freud.common.convert_array(
                d, 1, dtype=np.float32, contiguous=True,
                array_name="d")

        area = freud.common.convert_array(
                area, 1, dtype=np.float32, contiguous=True,
                array_name="area")

        if norms.shape[0] != facet_offs.shape[0] - 1:
            raise RuntimeError(
                ('Length of norms should be equal to number of facet offsets'
                    '- 1'))
        if d.shape[0] != facet_offs.shape[0] - 1:
            raise RuntimeError(
                ('Length of facet distances should be equal to number of facet'
                    'offsets - 1'))
        if area.shape[0] != facet_offs.shape[0] - 1:
            raise RuntimeError(
                ('Length of areas should be equal to number of facet offsets'
                    '- 1'))
        volume = float(volume)
        cdef np.ndarray[float, ndim = 2] cverts = verts
        cdef np.ndarray[unsigned int] cfacet_offs = facet_offs
        cdef np.ndarray[unsigned int] cfacets = facets
        cdef np.ndarray[float, ndim = 2] cnorms = norms
        cdef np.ndarray[float] cd = d
        cdef np.ndarray[float] carea = area
        self.thisptr.set_params(
                verts.shape[0],
                < vec3[float]*>cverts.data,
                facet_offs.shape[0] - 1,
                < unsigned int*>cfacet_offs.data,
                < unsigned int*>cfacets.data,
                < vec3[float]*>cnorms.data,
                < float*>cd.data,
                < float*>carea.data,
                volume)

    def set_rq(self, position, orientation):
        """Set particle positions and orientations.

        :param position: particle position vectors
        :param orientation: particle orientation quaternions
        :type position: :class:`numpy.ndarray`,
                        shape=(:math:`N_{particles}`, 3),
                        dtype= :class:`numpy.float32`
        :type orientation: :class:`numpy.ndarray`,
                            shape=(:math:`N_{particles}`, 4),
                            dtype= :class:`numpy.float32`
        """
        position = freud.common.convert_array(
                position, 2, dtype=np.float32, contiguous=True,
                array_name="position")
        if position.shape[1] != 3:
            raise TypeError('position should be an Nx3 array')

        orientation = freud.common.convert_array(
                orientation, 2, dtype=np.float32, contiguous=True,
                array_name="orientation")
        if orientation.shape[1] != 4:
            raise TypeError('orientation should be an Nx4 array')

        if position.shape[0] != orientation.shape[0]:
            raise TypeError(
                'position and orientation should have the same length')
        Np = position.shape[0]
        cdef np.ndarray[float, ndim = 2] cr = position
        cdef np.ndarray[float, ndim = 2] cq = orientation
        self.thisptr.set_rq(
                Np,
                < vec3[float]*>cr.data,
                < quat[float]*> cq.data)

    def set_density(self, float complex density):
        """Set scattering density.

        :param density: complex value of scattering density
        :type density: float complex
        """
        self.thisptr.set_density(density)
