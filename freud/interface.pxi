
from freud.util._VectorMath cimport vec3
cimport freud._interface as interface
from cython.operator cimport dereference
import numpy as np
cimport numpy as np

cdef class InterfaceMeasure:
    cdef interface.InterfaceMeasure *thisptr
    def __init__(self, box, float r_cut):
        cdef trajectory.Box cBox = trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new interface.InterfaceMeasure(cBox, r_cut)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, ref_points, points):
        ref_points = np.ascontiguousarray(ref_points, dtype=np.float32)
        if ref_points.ndim != 2 or ref_points.shape[1] != 3:
            raise RuntimeError('Need a list of 3D reference points for computeCellList()')
        points = np.ascontiguousarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise RuntimeError('Need a list of 3D points for computeCellList()')
        cdef np.ndarray cRef_points = ref_points
        cdef unsigned int Nref = ref_points.shape[0]
        cdef np.ndarray cPoints = points
        cdef unsigned int Np = points.shape[0]
        return self.thisptr.compute(<vec3[float]*> cRef_points.data, Nref, <vec3[float]*> cPoints.data, Np)
