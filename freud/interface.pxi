# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

from freud.util._VectorMath cimport vec3
cimport freud._interface as interface
cimport freud._box as _box
from cython.operator cimport dereference
import numpy as np
cimport numpy as np

cdef class InterfaceMeasure:
    """Measures the interface between two sets of points.

    .. moduleauthor:: Matthew Spellings <mspells@umich.edu>

    :param box: simulation box
    :param r_cut: Distance to search for particle neighbors
    :type box: :py:class:`freud.box.Box`
    :type r_cut: float
    """
    cdef interface.InterfaceMeasure * thisptr
    cdef box
    cdef rmax

    def __cinit__(self, box, float r_cut):
        cdef _box.Box cBox = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new interface.InterfaceMeasure(cBox, r_cut)
        self.box = box
        self.rmax = r_cut

    def __dealloc__(self):
        del self.thisptr

    def compute(self, ref_points, points, nlist=None):
        """Compute and return the number of particles at the interface between
        the two given sets of points.

        :param ref_points: one set of particle positions
        :param points: other set of particle positions
        :param nlist: :py:class:`freud.locality.NeighborList` object to use to find bonds
        :type ref_points: :class:`numpy.ndarray`, shape=(:math:`N_{particles}`, 3), dtype= :class:`numpy.float32`
        :type points: :class:`numpy.ndarray`, shape=(:math:`N_{particles}`, 3), dtype= :class:`numpy.float32`
        :type nlist: :py:class:`freud.locality.NeighborList`
        """
        ref_points = freud.common.convert_array(ref_points, 2, dtype=np.float32, contiguous=True,
                                                dim_message="ref_points must be a 2 dimensional array")
        points = freud.common.convert_array(points, 2, dtype=np.float32, contiguous=True,
                                            dim_message="points must be a 2 dimensional array")
        if ref_points.shape[1] != 3 or points.shape[1] != 3:
            raise RuntimeError('Need to provide array with x, y, z positions')

        defaulted_nlist = make_default_nlist(
            self.box, ref_points, points, self.rmax, nlist, None)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        cdef np.ndarray cRef_points = ref_points
        cdef unsigned int n_ref = ref_points.shape[0]
        cdef np.ndarray cPoints = points
        cdef unsigned int Np = points.shape[0]
        return self.thisptr.compute(nlist_ptr, < vec3[float]*> cRef_points.data, n_ref, < vec3[float]*> cPoints.data, Np)
