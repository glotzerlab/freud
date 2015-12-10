
from freud.util._VectorMath cimport vec3
cimport freud._locality as locality
from cython.operator cimport dereference
import numpy as np
cimport numpy as np

cdef class IteratorLinkCell:
    """Iterates over the particles in a cell.

    Example::

       # grab particles in cell 0
       for j in linkcell.itercell(0):
           print(positions[j])
    """
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
    """Supports efficiently finding all points in a set within a certain
    distance from a given point.

    :param box: :py:class:`freud.trajectory.Box` object
    :param cell_width: Maximum distance to find particles within

    .. note::

       :py:class:`freud.locality.LinkCell` supports 2D boxes; in this case, make sure to set the z coordinate of all points to 0.

    Example::

       # assume we have position as Nx3 array
       lc = LinkCell(box, 1.5)
       lc.computeCellList(box, positions)
       for i in range(positions.shape[0]):
           # cell containing particle i
           cell = lc.getCell(positions[0])
           # list of cell's neighboring cells
           cellNeighbors = lc.getCellNeighbors(cell)
           # iterate over neighboring cells (including our own)
           for neighborCell in cellNeighbors:
               # iterate over particles in each neighboring cell
               for neighbor in lc.itercell(neighborCell):
                   pass # do something with neighbor index
    """
    cdef locality.LinkCell *thisptr

    def __cinit__(self, box, cell_width):
        cdef trajectory.Box cBox = trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new locality.LinkCell(cBox, float(cell_width))

    def getBox(self):
        """Return the stored :py:class:`freud.trajectory.Box` object"""
        return BoxFromCPP(self.thisptr.getBox())

    def getNumCells(self):
        """Return the total number of cells for the current box."""
        return self.thisptr.getNumCells()

    def getCell(self, point):
        """Returns the index of the cell containing the given point

        :param point: array-like object of length 3 specifying the point coordinates
        """
        cdef float[:] cPoint = np.ascontiguousarray(point, dtype=np.float32)
        if len(cPoint) != 3:
            raise RuntimeError('Need a 3D point for getCell()')

        return self.thisptr.getCell(dereference(<vec3[float]*>&cPoint[0]))

    def itercell(self, unsigned int cell):
        """Return an iterator over all particles in the given cell

        :param cell: Cell index
        """
        result = IteratorLinkCell()
        cdef locality.IteratorLinkCell cResult = self.thisptr.itercell(cell)
        result.copy(cResult)
        return iter(result)

    def getCellNeighbors(self, cell):
        """Returns the neighboring cell indices of the given cell

        :param cell: Cell index
        """
        neighbors = self.thisptr.getCellNeighbors(int(cell))
        result = np.zeros(neighbors.size(), dtype=np.uint32)
        for i in range(neighbors.size()):
            result[i] = neighbors[i]
        return result

    def computeCellList(self, box, points):
        """Update the data structure for the given set of points

        :param box: :py:class:`freud.trajectory.Box` object
        :param points: Nx3 array-like object specifying coordinates
        """
        points = np.ascontiguousarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise RuntimeError('Need a list of 3D points for computeCellList()')
        cdef _trajectory.Box cBox = _trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        cdef np.ndarray cPoints = points
        cdef unsigned int Np = points.shape[0]
        with nogil:
            self.thisptr.computeCellList(cBox, <vec3[float]*> cPoints.data, Np)

cdef class NearestNeighbors:
    """Supports efficiently finding the N nearest neighbors of each point
    in a set for some fixed integer N.

    :param rmax: Initial guess of a distance to search within to find N neighbors
    :param n_neigh: Number of neighbors to find for each point
    """
    cdef locality.NearestNeighbors *thisptr

    def __cinit__(self, rmax, n_neigh):
        self.thisptr = new locality.NearestNeighbors(float(rmax), int(n_neigh))

    def getBox(self):
        """Return the stored :py:class:`freud.trajectory.Box` object"""
        return BoxFromCPP(self.thisptr.getBox())

    def getNNeigh(self):
        """Return the number of neighbors this object will find"""
        return self.thisptr.getNNeigh()

    def setRMax(self, float rmax):
        """Update the neighbor search distance guess"""
        self.thisptr.setRMax(rmax)

    def getRMax(self):
        """Return the current neighbor search distance guess"""
        return self.thisptr.getRMax()

    def getNeighbors(self, unsigned int i):
        """Return the N nearest neighbors of the reference point with index i

        :param i: index of the reference point to fetch the neighboring points of
        """
        cdef unsigned int nNeigh = self.thisptr.getNNeigh()
        result = np.zeros(nNeigh, dtype=np.uint32)
        cdef unsigned int start_idx = i*nNeigh
        cdef unsigned int *neighbors = self.thisptr.getNeighborList().get()
        for j in range(nNeigh):
            result[j] = neighbors[start_idx + j]

        return result

    def compute(self, box, ref_points, points):
        """Update the data structure for the given set of points

        :param box: :py:class:`freud.trajectory.Box` object
        :param ref_points: Reference points to find neighbors of
        :param points: Points to find as neighbors
        """
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
