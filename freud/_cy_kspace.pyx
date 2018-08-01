# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from . import common
import numpy as np

from .util._VectorMath cimport vec3, quat
from . cimport _kspace as kspace
from libcpp.memory cimport shared_ptr
from libcpp.complex cimport complex
from cython.operator cimport dereference

cimport numpy as np


# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class FTdelta:
    """Compute the Fourier transform of a set of delta peaks at a list of
    :math:`K` points.

    .. moduleauthor:: Jens Glaser <jsglaser@umich.edu>

    Attributes:
        FT (:class:`np.ndarray`): The Fourier transform.
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
        """Perform transform and store result internally."""
        self.thisptr.compute()
        return self

    def getFT(self):
        """ """
        cdef(float complex) * ft_points = self.thisptr.getFT().get()
        if not self.NK:
            return np.array([[]], dtype=np.complex64)
        result = np.zeros([self.NK], dtype=np.complex64)
        cdef float complex[:] flatBuffer = <float complex[:self.NK]> ft_points
        result.flat[:] = flatBuffer
        return result

    def set_K(self, K):
        """Set the :math:`K` values to evaluate.

        Args:
            K((:math:`N_{K}`, 3) :class:`numpy.ndarray`):
                :math:`K` values to evaluate.
        """
        K = common.convert_array(
            K, 2, dtype=np.float32, contiguous=True, array_name="K")
        if K.shape[1] != 3:
            raise TypeError('K should be an Nx3 array')

        self.NK = K.shape[0]
        # cdef unsigned int NK = <unsigned int> K.shape[0]
        cdef np.ndarray[float, ndim=2] cK = K
        self.thisptr.set_K(<vec3[float]*> cK.data, self.NK)

    def set_rq(self, position, orientation):
        """Set particle positions and orientations.

        Args:
            position ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Particle position vectors.
            orientation ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`):
                Particle orientation quaternions.
        """
        position = common.convert_array(
            position, 2, dtype=np.float32, contiguous=True,
            array_name="position")
        orientation = common.convert_array(
            orientation, 2, dtype=np.float32, contiguous=True,
            array_name="orientation")
        if position.shape[1] != 3:
            raise TypeError('position should be an Nx3 array')
        if orientation.shape[1] != 4:
            raise TypeError('orientation should be an Nx4 array')
        if position.shape[0] != orientation.shape[0]:
            raise TypeError(
                'position and orientation should have the same length')
        cdef unsigned int Np = <unsigned int> position.shape[0]
        cdef np.ndarray[float, ndim=2] cr = position
        cdef np.ndarray[float, ndim=2] cq = orientation
        self.thisptr.set_rq(Np, <vec3[float]*> cr.data, <quat[float]*> cq.data)

    def set_density(self, float complex density):
        """Set scattering density.

        Args:
            density (float complex): Complex value of scattering density.
        """
        self.thisptr.set_density(density)

cdef class FTsphere:
    """
    .. moduleauthor:: Jens Glaser <jsglaser@umich.edu>

    Attributes:
        FT (:class:`np.ndarray`): The Fourier transform.
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
        """Perform transform and store result internally."""
        self.thisptr.compute()
        return self

    def getFT(self):
        """ """
        cdef(float complex) * ft_points = self.thisptr.getFT().get()
        if not self.NK:
            return np.array([[]], dtype=np.complex64)
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.NK
        result = np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_COMPLEX64, ft_points)
        return result

    def set_K(self, K):
        """Set the :math:`K` values to evaluate.

        Args:
            K((:math:`N_{K}`, 3) :class:`numpy.ndarray`):
                :math:`K` values to evaluate.
        """
        K = np.ascontiguousarray(K, dtype=np.float32)
        if K.ndim != 2 or K.shape[1] != 3:
            raise TypeError('K should be an Nx3 array')
        self.NK = K.shape[0]
        cdef np.ndarray[float, ndim=2] cK = K
        self.thisptr.set_K(<vec3[float]*> cK.data, self.NK)

    def set_rq(self, position, orientation):
        """Set particle positions and orientations.

        Args:
            position ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Particle position vectors.
            orientation ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`):
                Particle orientation quaternions.
        """
        position = common.convert_array(
            position, 2, dtype=np.float32, contiguous=True,
            array_name="position")
        orientation = common.convert_array(
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
        cdef np.ndarray[float, ndim=2] cr = position
        cdef np.ndarray[float, ndim=2] cq = orientation
        self.thisptr.set_rq(Np, <vec3[float]*> cr.data, <quat[float]*> cq.data)

    def set_density(self, float complex density):
        """Set scattering density.

        Args:
            density (float complex): Complex value of scattering density.
        """
        self.thisptr.set_density(density)

    def set_radius(self, float radius):
        """Set particle volume according to radius.

        Args:
            radius (float): Particle radius.
        """
        self.thisptr.set_radius(radius)

cdef class FTpolyhedron:
    """
    .. moduleauthor:: Jens Glaser <jsglaser@umich.edu>

    Attributes:
        FT (:class:`np.ndarray`): The Fourier transform.
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
        """Perform transform and store result internally."""
        self.thisptr.compute()
        return self

    def getFT(self):
        """ """
        cdef(float complex) * ft_points = self.thisptr.getFT().get()
        if not self.NK:
            return np.array([[]], dtype=np.complex64)
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.NK
        result = np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_COMPLEX64, ft_points)
        return result

    def set_K(self, K):
        """Set the :math:`K` values to evaluate.

        Args:
            K ((:math:`N_{K}`, 3) :class:`numpy.ndarray`):
                :math:`K` values to evaluate.
        """
        K = common.convert_array(
            K, 2, dtype=np.float32, contiguous=True, array_name="K")
        if K.shape[1] != 3:
            raise TypeError('K should be an Nx3 array')
        self.NK = K.shape[0]
        cdef np.ndarray[float, ndim=2] cK = K
        self.thisptr.set_K(<vec3[float]*> cK.data, self.NK)

    def set_params(self, verts, facet_offs, facets, norms, d, area, volume):
        """Set polyhedron geometry.

        Args:
            verts ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Vertex coordinates.
            facet_offs ((:math:`N_{facets}`) :class:`numpy.ndarray`):
                Facet start offsets.
            facets ((:math:`N_{facets}`, 3) :class:`numpy.ndarray`):
                Facet vertex indices.
            norms ((:math:`N_{facets}`, 3) :class:`numpy.ndarray`):
                Facet normals.
            d ((:math:`N_{facets}`) :class:`numpy.ndarray`):
                Facet distances.
            area ((:math:`N_{facets}`) :class:`numpy.ndarray`):
                Facet areas.
            volume (float):
                Polyhedron volume.
        """
        verts = common.convert_array(
            verts, 2, dtype=np.float32, contiguous=True, array_name="verts")
        if verts.shape[1] != 3:
            raise TypeError('verts should be an Nx3 array')

        facet_offs = common.convert_array(
            facet_offs, 1, dtype=np.uint32, contiguous=True,
            array_name="facet_offs")

        facets = common.convert_array(
            facets, 1, dtype=np.uint32, contiguous=True, array_name="facets")

        norms = common.convert_array(
            norms, 2, dtype=np.float32, contiguous=True, array_name="norms")
        if norms.shape[1] != 3:
            raise TypeError('norms should be an Nx3 array')

        d = common.convert_array(
            d, 1, dtype=np.float32, contiguous=True, array_name="d")

        area = common.convert_array(
            area, 1, dtype=np.float32, contiguous=True, array_name="area")

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
        cdef np.ndarray[float, ndim=2] cverts = verts
        cdef np.ndarray[unsigned int] cfacet_offs = facet_offs
        cdef np.ndarray[unsigned int] cfacets = facets
        cdef np.ndarray[float, ndim=2] cnorms = norms
        cdef np.ndarray[float] cd = d
        cdef np.ndarray[float] carea = area
        self.thisptr.set_params(
            verts.shape[0],
            <vec3[float]*> cverts.data,
            facet_offs.shape[0] - 1,
            <unsigned int*> cfacet_offs.data,
            <unsigned int*> cfacets.data,
            <vec3[float]*> cnorms.data,
            <float*> cd.data,
            <float*> carea.data,
            volume)

    def set_rq(self, position, orientation):
        """Set particle positions and orientations.

        Args:
            position ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Particle position vectors.
            orientation ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`):
                Particle orientation quaternions.
        """
        position = common.convert_array(
            position, 2, dtype=np.float32, contiguous=True,
            array_name="position")
        if position.shape[1] != 3:
            raise TypeError('position should be an Nx3 array')

        orientation = common.convert_array(
            orientation, 2, dtype=np.float32, contiguous=True,
            array_name="orientation")
        if orientation.shape[1] != 4:
            raise TypeError('orientation should be an Nx4 array')

        if position.shape[0] != orientation.shape[0]:
            raise TypeError(
                'position and orientation should have the same length')
        Np = position.shape[0]
        cdef np.ndarray[float, ndim=2] cr = position
        cdef np.ndarray[float, ndim=2] cq = orientation
        self.thisptr.set_rq(
            Np,
            <vec3[float]*> cr.data,
            <quat[float]*> cq.data)

    def set_density(self, float complex density):
        """Set scattering density.

        Args:
            density (float complex): Complex value of scattering density.
        """
        self.thisptr.set_density(density)
