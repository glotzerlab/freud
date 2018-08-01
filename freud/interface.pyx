# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The interface module contains functions to measure the interface between sets
of points.
"""

from . import common
import numpy as np

from .util._VectorMath cimport vec3
from cython.operator cimport dereference
from .locality cimport NeighborList
from .locality import make_default_nlist, make_default_nlist_nn

from . cimport _interface, _box, _locality

cimport numpy as np

cdef class InterfaceMeasure:
    """Measures the interface between two sets of points.

    .. moduleauthor:: Matthew Spellings <mspells@umich.edu>

    Args:
        box (:py:class:`freud.box.Box`): Simulation box.
        r_cut (float): Distance to search for particle neighbors.
    """
    cdef _interface.InterfaceMeasure * thisptr
    cdef box
    cdef rmax

    def __cinit__(self, box, float r_cut):
        box = common.convert_box(box)
        cdef _box.Box cBox = _box.Box(
            box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
            box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new _interface.InterfaceMeasure(cBox, r_cut)
        self.box = box
        self.rmax = r_cut

    def __dealloc__(self):
        del self.thisptr

    def compute(self, ref_points, points, nlist=None):
        """Compute and return the number of particles at the interface between
        the two given sets of points.

        Args:
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                One set of particle positions.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Other set of particle positions.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        ref_points = common.convert_array(
            ref_points, 2, dtype=np.float32, contiguous=True,
            array_name="ref_points")
        points = common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if ref_points.shape[1] != 3 or points.shape[1] != 3:
            raise RuntimeError('Need to provide array with x, y, z positions')

        defaulted_nlist = make_default_nlist(
            self.box, ref_points, points, self.rmax, nlist, None)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef _locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        cdef np.ndarray cRef_points = ref_points
        cdef unsigned int n_ref = ref_points.shape[0]
        cdef np.ndarray cPoints = points
        cdef unsigned int Np = points.shape[0]
        return self.thisptr.compute(
            nlist_ptr,
            <vec3[float]*> cRef_points.data,
            n_ref,
            <vec3[float]*> cPoints.data,
            Np)
