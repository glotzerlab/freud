
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
    .. moduleauthor:: Ben Schultz <baschult@umich.edu@umich.edu>
    """
    cdef voronoi.VoronoiBuffer *thisptr

    def __cinit__(self, box):
        cdef _box.Box cBox = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new voronoi.VoronoiBuffer(cBox)

    def compute(self, points, float buffer):
        points = np.ascontiguousarray(points, dtype=np.float32)
        dimensions = 2 if self.thisptr.getBox().is2D() else 3
        if points.ndim != 2 or points.shape[1] != dimensions:
            raise RuntimeError('Need a list of {}D points for VoronoiBuffer.compute()'.format(dimensions))
        cdef np.ndarray cPoints = points
        cdef unsigned int Np = points.shape[0]
        self.thisptr.compute(<float3*> cPoints.data, Np, buffer)

    def getBufferParticles(self):
        cdef _box.Box cBox = self.thisptr.getBox()
        cdef unsigned int buffer_size = dereference(self.thisptr.getBufferParticles().get()).size()
        cdef float3* buffer_points = &dereference(self.thisptr.getBufferParticles().get())[0]
        if not buffer_size:
            return np.array([[]], dtype=np.float32)
        shape = [buffer_size, 3]
        result = np.zeros(shape, dtype=np.float32)
        cdef float[:] flatBuffer = <float[:shape[0]*shape[1]]> (<float*> buffer_points)
        result.flat[:] = flatBuffer
        if cBox.is2D():
            return result[:, :2]
        else:
            return result
