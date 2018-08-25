# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The particle buffer helps to efficiently replicate particle positions across
periodic boundaries of a system.
"""

import numpy as np
import logging
import copy
import freud.common

from libcpp.vector cimport vector
from freud.util._VectorMath cimport vec3
from freud.util cimport _ParticleBuffer
from cython.operator cimport dereference

cimport freud.box
cimport freud.locality
cimport numpy as np


logger = logging.getLogger(__name__)

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class ParticleBuffer:
    """Replicates particles up to a buffer distance outside the box from
    periodic images.

    .. moduleauthor:: Ben Schultz <baschult@umich.edu>
    .. moduleauthor:: Bradley Dice <bdice@bradleydice.com>

    Args:
        box (py:class:`freud.box.Box`): Simulation box.

    Attributes:
        buffer_particles (:class:`numpy.ndarray`):
            The buffer particles.
        buffer_ids (:class:`numpy.ndarray`):
            The buffer ids.
    """
    def __cinit__(self, box):
        cdef freud.box.Box b = freud.common.convert_box(box)
        self.thisptr = new _ParticleBuffer.ParticleBuffer(
            dereference(b.thisptr))

    def compute(self, points, float buffer):
        """Compute the particle buffer.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points used to calculate particle buffer.
            buffer (float):
                Buffer distance within which to look for images.
        """
        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name='points')

        if points.shape[1] != 3:
            raise RuntimeError(
                'Need a list of 3D points for ParticleBuffer.compute()')
        cdef np.ndarray cPoints = points
        cdef unsigned int Np = points.shape[0]
        self.thisptr.compute(<vec3[float]*> cPoints.data, Np, buffer)
        return self

    @property
    def buffer_particles(self):
        cdef unsigned int buffer_size = \
            dereference(self.thisptr.getBufferParticles().get()).size()
        cdef vec3[float] * buffer_points = \
            &dereference(self.thisptr.getBufferParticles().get())[0]
        if not buffer_size:
            return np.array([[]], dtype=np.float32)

        cdef vector[vec3[float]]*bufferPar = \
            self.thisptr.getBufferParticles().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = buffer_size
        nbins[1] = 3

        cdef np.ndarray[float, ndim=2] result = \
            np.PyArray_SimpleNewFromData(2, nbins, np.NPY_FLOAT32,
                                         <void*> dereference(bufferPar).data())

        return result

    def getBufferParticles(self):
        warnings.warn("The getBufferParticles function is deprecated in favor "
                      "of the buffer_particles class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.buffer_particles

    @property
    def buffer_ids(self):
        cdef unsigned int buffer_size = \
            dereference(self.thisptr.getBufferParticles().get()).size()
        cdef unsigned int * buffer_ids = \
            &dereference(self.thisptr.getBufferIds().get())[0]
        if not buffer_size:
            return np.array([[]], dtype=np.uint32)

        cdef vector[unsigned int]*bufferIds = self.thisptr.getBufferIds().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = buffer_size

        cdef np.ndarray[unsigned int, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_UINT32,
                                         <void*> dereference(bufferIds).data())

        return result

    def getBufferIds(self):
        warnings.warn("The getBufferIds function is deprecated in favor "
                      "of the buffer_ids class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.buffer_ids
