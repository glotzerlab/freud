# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

import numpy as np
from cython.view cimport array as cvarray
from libcpp.vector cimport vector
from freud.util._VectorMath cimport vec3
from cython.operator cimport dereference
cimport freud._voronoi as voronoi
cimport freud._box as _box
cimport numpy as np

cdef class VoronoiBuffer:
    """
    .. moduleauthor:: Ben Schultz <baschult@umich.edu>
    .. moduleauthor:: Bradley Dice <bdice@bradleydice.com>
    """
    cdef voronoi.VoronoiBuffer * thisptr

    def __cinit__(self, box):
        box = freud.common.convert_box(box)
        cdef _box.Box cBox = _box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new voronoi.VoronoiBuffer(cBox)

    def compute(self, points, float buffer):
        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                dim_message='points must be a 3 dimensional array')

        if points.shape[1] != 3:
            raise RuntimeError(
                'Need a list of 3D points for VoronoiBuffer.compute()')
        cdef np.ndarray cPoints = points
        cdef unsigned int Np = points.shape[0]
        self.thisptr.compute(< vec3[float]*> cPoints.data, Np, buffer)
        return self

    def getBufferParticles(self):
        cdef unsigned int buffer_size = dereference(
                self.thisptr.getBufferParticles().get()).size()
        cdef vec3[float] * buffer_points = &dereference(
                self.thisptr.getBufferParticles().get())[0]
        if not buffer_size:
            return np.array([[]], dtype=np.float32)

        cdef vector[vec3[float]]*bufferPar = self.thisptr.getBufferParticles().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = buffer_size
        nbins[1] = 3

        cdef np.ndarray[float, ndim = 2] result = \
                np.PyArray_SimpleNewFromData(
                    2, nbins, np.NPY_FLOAT32,
                    <void*> dereference(bufferPar).data())

        return result

    def getBufferIds(self):
        cdef unsigned int buffer_size = dereference(
                self.thisptr.getBufferParticles().get()).size()
        cdef unsigned int * buffer_ids = &dereference(
                self.thisptr.getBufferIds().get())[0]
        if not buffer_size:
            return np.array([[]], dtype=np.uint32)

        cdef vector[unsigned int]*bufferIds = self.thisptr.getBufferIds().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = buffer_size

        cdef np.ndarray[unsigned int, ndim = 1] result = \
                np.PyArray_SimpleNewFromData(
                    1, nbins, np.NPY_UINT32,
                    <void*> dereference(bufferIds).data())

        return result
