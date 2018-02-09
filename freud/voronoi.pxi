# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

from cython.view cimport array as cvarray
from libcpp.vector cimport vector
from freud.util._cudaTypes cimport float3
cimport freud._voronoi as voronoi
cimport freud._box as _box
from cython.operator cimport dereference
import numpy as np
cimport numpy as np

cdef class VoronoiBuffer:
    """
    .. moduleauthor:: Ben Schultz <baschult@umich.edu>
    """
    cdef voronoi.VoronoiBuffer * thisptr

    def __cinit__(self, box):
        cdef _box.Box cBox = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new voronoi.VoronoiBuffer(cBox)

    def compute(self, points, float buffer):
        points = freud.common.convert_array(points, 2, dtype=np.float32, contiguous=True,
                                            dim_message='points must be a 3 dimensional array')

        if points.shape[1] != 3:
            raise RuntimeError(
                'Need a list of 3D points for VoronoiBuffer.compute()')
        cdef np.ndarray cPoints = points
        cdef unsigned int Np = points.shape[0]
        self.thisptr.compute(< float3*> cPoints.data, Np, buffer)
        return self

    def getBufferParticles(self):
        cdef _box.Box cBox = self.thisptr.getBox()
        cdef unsigned int buffer_size = dereference(self.thisptr.getBufferParticles().get()).size()
        cdef float3 * buffer_points = &dereference(self.thisptr.getBufferParticles().get())[0]
        if not buffer_size:
            return np.array([[]], dtype=np.float32)

        cdef vector[float3]*bufferPar = self.thisptr.getBufferParticles().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = buffer_size
        nbins[1] = 3

        cdef np.ndarray[float, ndim = 2] result = np.PyArray_SimpleNewFromData(2, nbins, np.NPY_FLOAT32, < void*>dereference(bufferPar).data())

        return result
