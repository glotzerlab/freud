
from freud.util._VectorMath cimport vec3
cimport freud._locality as locality
from cython.operator cimport dereference
import numpy as np
cimport numpy as np

cdef class IteratorLinkCell:
    cdef locality.IteratorLinkCell *thisptr

    def __cinit__(self):
        self.thisptr = new locality.IteratorLinkCell()

    cdef void copy(self, const locality.IteratorLinkCell &rhs):
        self.thisptr.copy(rhs)

    def next(self):
        cdef unsigned int result = self.thisptr.next()
        if self.thisptr.atEnd():
            raise StopIteration()
        return result

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

cdef class LinkCell:
    cdef locality.LinkCell *thisptr

    def __cinit__(self, box, cell_width):
        cdef trajectory.Box cBox = trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new locality.LinkCell(cBox, float(cell_width))

    def getBox(self):
        return BoxFromCPP(self.thisptr.getBox())

    def getNumCells(self):
        return self.thisptr.getNumCells()

    def getCell(self, point):
        cdef float[:] cPoint = np.ascontiguousarray(point, dtype=np.float32)
        if len(cPoint) != 3:
            raise RuntimeError('Need a 3D point for getCell()')

        return self.thisptr.getCell(dereference(<vec3[float]*>&cPoint[0]))

    def itercell(self, unsigned int cell):
        result = IteratorLinkCell()
        cdef locality.IteratorLinkCell cResult = self.thisptr.itercell(cell)
        result.copy(cResult)
        return iter(result)

    def getCellNeighbors(self, cell):
        neighbors = self.thisptr.getCellNeighbors(int(cell))
        result = np.zeros(neighbors.size(), dtype=np.uint32)
        for i in range(neighbors.size()):
            result[i] = neighbors[i]
        return result

    def computeCellList(self, box, points):
        points = np.ascontiguousarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise RuntimeError('Need a list of 3D points for computeCellList()')
        cdef _trajectory.Box cBox = _trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        cdef np.ndarray cPoints = points
        cdef unsigned int Np = points.shape[0]
        with nogil:
            self.thisptr.computeCellList(cBox, <vec3[float]*> cPoints.data, Np)

cdef class NearestNeighbors:
    cdef locality.NearestNeighbors *thisptr

    def __cinit__(self, rmax, n_neigh):
        self.thisptr = new locality.NearestNeighbors(float(rmax), int(n_neigh))

    def getBox(self):
        return BoxFromCPP(self.thisptr.getBox())

    def getNNeigh(self):
        return self.thisptr.getNNeigh()

    def setRMax(self, float rmax):
        self.thisptr.setRMax(rmax)

    def getRMax(self):
        return self.thisptr.getRMax()

    def getNeighbors(self, unsigned int i):
        cdef unsigned int nNeigh = self.thisptr.getNNeigh()
        result = np.zeros(nNeigh, dtype=np.uint32)
        cdef unsigned int start_idx = i*nNeigh
        cdef unsigned int *neighbors = self.thisptr.getNeighborList().get()
        for j in range(nNeigh):
            result[j] = neighbors[start_idx + j]

        return result

    def compute(self, box, ref_points, points):
        ref_points = np.ascontiguousarray(ref_points, dtype=np.float32)
        if ref_points.ndim != 2 or ref_points.shape[1] != 3:
            raise RuntimeError('Need a list of 3D reference points for computeCellList()')
        points = np.ascontiguousarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise RuntimeError('Need a list of 3D points for computeCellList()')
        cdef _trajectory.Box cBox = _trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        cdef np.ndarray cRef_points = ref_points
        cdef unsigned int Nref = ref_points.shape[0]
        cdef np.ndarray cPoints = points
        cdef unsigned int Np = points.shape[0]
        with nogil:
            self.thisptr.compute(cBox, <vec3[float]*> cRef_points.data, Nref, <vec3[float]*> cPoints.data, Np)
