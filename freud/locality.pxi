
import sys
from freud.util._VectorMath cimport vec3
cimport freud._locality as locality
cimport freud._box as _box;
from cython.operator cimport dereference
import numpy as np
cimport numpy as np

cdef class IteratorLinkCell:
    """Iterates over the particles in a cell.

    .. moduleauthor:: Joshua Anderson <joaander@umich.edu>

    Example::

       # grab particles in cell 0
       for j in linkcell.itercell(0):
           print(positions[j])
    """
    cdef locality.IteratorLinkCell *thisptr

    def __cinit__(self):
        # must be running python 3.x
        current_version = sys.version_info
        if current_version.major < 3:
            raise RuntimeError("Must use python 3.x or greater to use IteratorLinkCell")
        else:
            self.thisptr = new locality.IteratorLinkCell()

    def __dealloc__(self):
        del self.thisptr

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

    .. moduleauthor:: Joshua Anderson <joaander@umich.edu>

    :param box: :py:class:`freud._box.Box` object
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
        cdef _box.Box cBox = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new locality.LinkCell(cBox, float(cell_width))

    def __dealloc__(self):
        del self.thisptr

    def getBox(self):
        """Return the stored :py:class:`freud._box.Box` object"""
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
        current_version = sys.version_info
        if current_version.major < 3:
            raise RuntimeError("Must use python 3.x or greater to use itercell")
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

        :param box: :py:class:`freud._box.Box` object
        :param points: Nx3 array-like object specifying coordinates
        """
        points = np.ascontiguousarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise RuntimeError('Need a list of 3D points for computeCellList()')
        cdef _box.Box cBox = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        cdef np.ndarray cPoints = points
        cdef unsigned int Np = points.shape[0]
        with nogil:
            self.thisptr.computeCellList(cBox, <vec3[float]*> cPoints.data, Np)

cdef class NearestNeighbors:
    """Supports efficiently finding the N nearest neighbors of each point
    in a set for some fixed integer N.

    - strict_cut = True: rmax will be strictly obeyed, and any particle which has fewer than N neighbors will have \
        values of UINT_MAX assigned
    - strict_cut = False: rmax will be expanded to find requested number of neighbors. If rmax increases to the \
        point that a cell list cannot be constructed, a warning will be raised and neighbors found will be returned

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    :param rmax: Initial guess of a distance to search within to find N neighbors
    :param n_neigh: Number of neighbors to find for each point
    :param scale: multiplier by which to automatically increase rmax value by if requested number of neighbors is not \
        found. Only utilized if strict_cut is False. Scale must be greater than 1
    :param strict_cut: whether to use a strict rmax or allow for automatic expansion
    :type rmax: float
    :type n_neigh: int
    :type scale: float
    :type strict_cut: bool
    """
    cdef locality.NearestNeighbors *thisptr

    def __cinit__(self, rmax, n_neigh, scale=1.1, strict_cut=False):
        if scale < 1:
            raise RuntimeError("scale must be greater than 1")
        self.thisptr = new locality.NearestNeighbors(float(rmax), int(n_neigh), float(scale), bool(strict_cut))

    def __dealloc__(self):
        del self.thisptr

    def getBox(self):
        """Return the stored :py:class:`freud._box.Box` object"""
        return BoxFromCPP(self.thisptr.getBox())

    def getNumNeighbors(self):
        """Return the number of neighbors this object will find"""
        return self.thisptr.getNumNeighbors()

    def setRMax(self, float rmax):
        """Update the neighbor search distance guess"""
        self.thisptr.setRMax(rmax)

    def setCutMode(self, strict_cut):
        """
        Set mode to handle rmax by Nearest Neighbors.

        - strict_cut = True: rmax will be strictly obeyed, and any particle which has fewer than N neighbors will have \
            values of UINT_MAX assigned
        - strict_cut = False: rmax will be expanded to find requested number of neighbors. If rmax increases to the \
            point that a cell list cannot be constructed, a warning will be raised and neighbors found will be returned

        :param strict_cut: whether to use a strict rmax or allow for automatic expansion
        :type strict_cut: bool
        """
        self.thisptr.setCutMode(strict_cut)

    def getRMax(self):
        """Return the current neighbor search distance guess"""
        return self.thisptr.getRMax()

    def getNeighbors(self, unsigned int i):
        """Return the N nearest neighbors of the reference point with index i

        :param i: index of the reference point to fetch the neighboring points of
        """
        cdef unsigned int nNeigh = self.thisptr.getNumNeighbors()
        result = np.zeros(nNeigh, dtype=np.uint32)
        cdef unsigned int start_idx = i*nNeigh
        cdef unsigned int *neighbors = self.thisptr.getNeighborList().get()
        for j in range(nNeigh):
            result[j] = neighbors[start_idx + j]

        return result

    def getRsq(self, unsigned int i):
        """
        Return the Rsq values for the N nearest neighbors of the reference point with index i

        :param i: index of the reference point of which to fetch the neighboring point distances
        :type i: unisigned int
        :return: squared distances of the N nearest neighbors
        :rtype: np.ndarray(shape=N, dtype=np.float32)
        """
        pass

    def getRsqList(self):
        """
        """

    def compute(self, box, ref_points, points):
        """Update the data structure for the given set of points

        :param box: :py:class:`freud._box.Box` object
        :param ref_points: Reference points to find neighbors of
        :param points: Points to find as neighbors
        """
        ref_points = np.ascontiguousarray(ref_points, dtype=np.float32)
        if ref_points.ndim != 2 or ref_points.shape[1] != 3:
            raise RuntimeError('Need a list of 3D reference points for computeCellList()')
        points = np.ascontiguousarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise RuntimeError('Need a list of 3D points for computeCellList()')
        cdef _box.Box cBox = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        cdef np.ndarray cRef_points = ref_points
        cdef unsigned int n_ref = ref_points.shape[0]
        cdef np.ndarray cPoints = points
        cdef unsigned int Np = points.shape[0]
        with nogil:
            self.thisptr.compute(cBox, <vec3[float]*> cRef_points.data, n_ref, <vec3[float]*> cPoints.data, Np)
