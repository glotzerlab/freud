from freud.util._VectorMath cimport vec3
cimport freud._interface as interface
cimport freud._box as _box;
from cython.operator cimport dereference
import numpy as np
cimport numpy as np

cdef class InterfaceMeasure:
    """Measures the interface between two sets of points.

    :param box: :py:class:`freud._box.Box` object
    :param r_cut: Distance to search for particle neighbors
    """
    cdef interface.InterfaceMeasure *thisptr

    def __cinit__(self, box, float r_cut):
        cdef _box.Box cBox = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new interface.InterfaceMeasure(cBox, r_cut)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, ref_points, points):
        """Compute and return the number of particles at the interface between
        the two given sets of points.

        :param ref_points: Nx3 array-like object of one set of points
        :param points: Nx3 array-like object of the other set of points
        """

        ref_points = np.ascontiguousarray(ref_points, dtype=np.float32)
        if ref_points.ndim != 2 or ref_points.shape[1] != 3:
            raise RuntimeError('Need a list of 3D reference points for computeCellList()')
        points = np.ascontiguousarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise RuntimeError('Need a list of 3D points for computeCellList()')
        cdef np.ndarray cRef_points = ref_points
        cdef unsigned int n_ref = ref_points.shape[0]
        cdef np.ndarray cPoints = points
        cdef unsigned int Np = points.shape[0]
        return self.thisptr.compute(<vec3[float]*> cRef_points.data, n_ref, <vec3[float]*> cPoints.data, Np)
