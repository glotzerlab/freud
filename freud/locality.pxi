# Copyright (c) 2010-2016 The Regents of the University of Michigan
# This file is part of the Freud project, released under the BSD 3-Clause License.

import sys
from libcpp cimport bool as cbool
from freud.util._VectorMath cimport vec3
cimport freud._locality as locality
cimport freud._box as _box;
from cython.operator cimport dereference
import numpy as np
cimport numpy as np

cdef class NeighborList:
    """Class representing a certain number of "bonds" between
    particles. Computation methods will iterate over these bonds when
    searching for neighboring particles.

    NeighborList objects are constructed for two sets of position
    arrays A (alternatively *reference points*; of length :math:`n_A`)
    and B (alternatively *target points*; of length :math:`n_B`) and
    hold a set of :math:`\left(i, j\\right): i < n_A, j < n_B` index
    pairs corresponding to near-neighbor points in A and B,
    respectively.

    For efficiency, all bonds for a particular reference particle i
    are contiguous and bonds are stored in order based on reference
    particle index i. The first bond index corresponding to a given
    particle can be found in :math:`log(n_{bonds})` time using
    :py:meth:`find_first_index`.

    .. moduleauthor:: Matthew Spellings <mspells@umich.edu>

    .. note::

       Typically, in python you will only manipulate a :py:class:`freud.locality.NeighborList` object that you receive from a neighbor search algorithm, such as :py:class:`freud.locality.LinkCell` and :py:class:`freud.locality.NearestNeighbors`.

    Example::

       # assume we have position as Nx3 array
       lc = LinkCell(box, 1.5).compute(box, positions)
       nlist = lc.nlist

       # get all vectors from central particles to their neighbors
       rijs = positions[nlist.index_j] - positions[nlist.index_i]
       box.wrap(rijs)
    """
    cdef locality.NeighborList *thisptr
    cdef char _managed

    @classmethod
    def from_arrays(cls, Nref, Ntarget, index_i, index_j, weights=None):
        """Create a NeighborList from a set of bond information arrays.

        :param Nref: Number of reference points (corresponding to index_i)
        :param Ntarget: Number of target points (corresponding to index_j)
        :param index_i: Array of integers corresponding to indices in the set of reference points
        :param index_j: Array of integers corresponding to indices in the set of target points
        :param weights: Array of per-bond weights (if None is given, use a value of 1 for each weight)
        :type Nref: unsigned int
        :type Ntarget: unsigned int
        :type index_i: Array-like of unsigned ints, length num_bonds
        :type index_j: Array-like of unsigned ints, length num_bonds
        :type weights: Array-like of floats, length num_bonds
        """
        index_i = np.asarray(index_i, dtype=np.uint64)
        index_j = np.asarray(index_j, dtype=np.uint64)

        if index_i.ndim != 1 or index_j.ndim != 1:
            raise TypeError('index_i and index_j should be a 1D arrays')
        if index_i.shape != index_j.shape:
            raise TypeError('index_i and index_j should be the same size')

        if weights is None:
            weights = np.ones(index_i.shape, dtype=np.float32)
        else:
            weights = np.asarray(weights, dtype=np.float32)
        if weights.shape != index_i.shape:
            raise TypeError('weights and index_i should be the same size')

        cdef size_t n_bonds = index_i.shape[0]
        cdef size_t c_Nref = Nref
        cdef size_t c_Ntarget = Ntarget
        cdef np.ndarray[size_t, ndim=1] c_index_i = index_i
        cdef np.ndarray[size_t, ndim=1] c_index_j = index_j
        cdef np.ndarray[float, ndim=1] c_weights = weights

        cdef size_t last_i
        cdef int i
        if n_bonds:
            last_i = c_index_i[0]
            i = last_i
            for bond in range(n_bonds):
                i = c_index_i[bond]
                if i < last_i:
                    raise RuntimeError('index_i is not sorted')
                if c_Nref <= i:
                    raise RuntimeError('Nref is too small for a value found in index_i')
                if c_Ntarget <= c_index_j[bond]:
                    raise RuntimeError('Ntarget is too small for a value found in index_j')
                last_i = i

        result = cls()
        cdef NeighborList c_result = result
        c_result.thisptr.resize(n_bonds)
        cdef size_t* c_neighbors_ptr = c_result.thisptr.getNeighbors()
        cdef float *c_weights_ptr = c_result.thisptr.getWeights()

        for bond in range(n_bonds):
            c_neighbors_ptr[2*bond] = c_index_i[bond]
            c_neighbors_ptr[2*bond + 1] = c_index_j[bond]
            c_weights_ptr[bond] = c_weights[bond]

        c_result.thisptr.setNumBonds(n_bonds, c_Nref, c_Ntarget)

        return result

    cdef refer_to(self, locality.NeighborList *other):
        """Makes this cython wrapper object point to a different C++ object,
        deleting the one we are already holding if necessary. We do not
        own the memory of the other C++ object."""
        if self._managed:
            del self.thisptr
        self._managed = False
        self.thisptr = other

    def __cinit__(self):
        self._managed = True
        self.thisptr = new locality.NeighborList()

    def __dealloc__(self):
        if self._managed:
            del self.thisptr

    cdef locality.NeighborList *get_ptr(self):
        """Returns a pointer to the raw C++ object we are wrapping"""
        return self.thisptr

    cdef void copy_c(self, NeighborList other):
        """Copies the contents of other into ourself"""
        self.thisptr.copy(dereference(other.thisptr))

    def copy(self, other=None):
        """Create a copy. If other is given, copy its contents into ourself;
        otherwise, return a copy of ourself."""
        if other is not None:
            assert isinstance(other, NeighborList)
            self.copy_c(other)
            return self
        else:
            new_copy = NeighborList()
            new_copy.copy(self)
            return new_copy

    @property
    def index_i(self):
        """Returns the reference point indices from the last set of points we
        were evaluated with. This array is read-only to prevent
        breakage of find_first_index."""
        cdef np.npy_intp size[2]
        size[0] = self.thisptr.getNumBonds()
        size[1] = 2
        cdef np.ndarray[np.uint64_t, ndim=2] result = np.PyArray_SimpleNewFromData(2, size, np.NPY_UINT64, <void*> self.thisptr.getNeighbors())
        result.flags.writeable = False
        return result[:, 0]

    @property
    def index_j(self):
        """Returns the target point indices from the last set of points we
        were evaluated with. This array is read-only to prevent
        breakage of find_first_index."""
        cdef np.npy_intp size[2]
        size[0] = self.thisptr.getNumBonds()
        size[1] = 2
        cdef np.ndarray[np.uint64_t, ndim=2] result = np.PyArray_SimpleNewFromData(2, size, np.NPY_UINT64, <void*> self.thisptr.getNeighbors())
        result.flags.writeable = False
        return result[:, 1]

    @property
    def weights(self):
        """Returns the per-bond weights from the last set of points we were
        evaluated with"""
        cdef np.npy_intp size[1]
        size[0] = self.thisptr.getNumBonds()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, size, np.NPY_FLOAT32, <void*> self.thisptr.getWeights())
        return result

    @property
    def segments(self):
        """Returns a *segment array*, which is an array of length `N_ref`
        indicating the first bond index for each reference particle from the
        last set of points we were evaluated with."""
        result = np.zeros((self.thisptr.getNumI(),), dtype=np.int64)
        cdef size_t* neighbors = self.thisptr.getNeighbors()
        cdef size_t last_i = -1
        cdef size_t i = -1
        for bond in range(self.thisptr.getNumBonds()):
            i = neighbors[2*bond]
            if i != last_i:
                result[i] = bond
            last_i = i

        return result

    @property
    def neighbor_counts(self):
        """Returns a *neighbor count array*, which is an array of length
        `N_ref` indicating the number of neighbors for each reference particle
        from the last set of points we were evaluated with."""
        result = np.zeros((self.thisptr.getNumI(),), dtype=np.int64)
        cdef size_t* neighbors = self.thisptr.getNeighbors()
        cdef size_t last_i = -1
        cdef size_t i = -1
        cdef size_t n = 0
        for bond in range(self.thisptr.getNumBonds()):
            i = neighbors[2*bond]
            if i != last_i and i > 0:
                result[i - 1] = n
                n = 0
            last_i = i
            n += 1
        result[-1] = n

        return result

    def __len__(self):
        """Returns the number of bonds stored in this object"""
        return self.thisptr.getNumBonds()

    def find_first_index(self, unsigned int i):
        """Returns the lowest bond index corresponding to a reference particle
        with index >=i"""
        return self.thisptr.find_first_index(i)

    def filter(self, filt):
        """Removes bonds that satisfy a boolean criterion.

        :param filt: Boolean-like array of bonds to keep (True => bond stays)

        .. note:: This method modifies this object in-place

        Example::

            # keep only the bonds between particles of type A and type B
            nlist.filter(types[nlist.index_i] != types[nlist.index_j])
        """
        filt = np.ascontiguousarray(filt, dtype=np.bool)
        cdef np.ndarray[np.uint8_t, ndim=1, cast=True] filt_c = filt
        cdef cbool *filt_ptr = <cbool*> filt_c.data
        self.thisptr.filter(filt_ptr)
        return self

    def filter_r(self, box, ref_points, points, float rmax, float rmin=0):
        """Removes bonds that are outside of a given radius range.

        :param ref_points: reference points to use for filtering
        :param points: target points to use for filtering
        :param rmax: maximum bond distance in the resulting neighbor list
        :param rmin: minimum bond distance in the resulting neighbor list

        .. note:: This method modifies this object in-place
        """
        ref_points = freud.common.convert_array(ref_points, 2, dtype=np.float32, contiguous=True,
            dim_message="ref_points must be a 2 dimensional array")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        points = freud.common.convert_array(points, 2, dtype=np.float32, contiguous=True,
            dim_message="points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        cdef _box.Box cBox = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        cdef np.ndarray cRef_points = ref_points
        cdef np.ndarray cPoints = points
        cdef size_t nRef = ref_points.shape[0]
        cdef size_t nP = points.shape[0]

        self.thisptr.validate(nRef, nP)
        self.thisptr.filter_r(cBox, <vec3[float]*> cRef_points.data, <vec3[float]*> cPoints.data, rmax, rmin)
        return self

def make_default_nlist(box, ref_points, points, rmax, nlist=None, exclude_ii=None):
    """Helper function to return a neighbor list object if is given, or to
    construct one using LinkCell if it is not."""
    if nlist is not None:
        return nlist, nlist

    cdef LinkCell lc = LinkCell(box, rmax).computeCellList(box, ref_points, points, exclude_ii)

    # return the owner of the neighbor list as well to prevent gc problems
    return lc.nlist, lc

def make_default_nlist_nn(box, ref_points, points, n_neigh, nlist=None, exclude_ii=None, rmax_guess=2.):
    """Helper function to return a neighbor list object if is given, or to
    construct one using NearestNeighbors if it is not."""
    if nlist is not None:
        return nlist, nlist

    cdef NearestNeighbors nn = NearestNeighbors(rmax_guess, n_neigh).compute(box, ref_points, points)

    # return the owner of the neighbor list as well to prevent gc problems
    return nn.nlist, nn

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

    :param box: simulation box
    :param cell_width: Maximum distance to find particles within
    :type box: :py:class:`freud.box.Box`
    :type cell_width: float

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

       # using NeighborList API
       dens = density.LocalDensity(1.5, 1, 1)
       dens.compute(box, positions, nlist=lc.nlist)
    """
    cdef locality.LinkCell *thisptr
    cdef NeighborList _nlist

    def __cinit__(self, box, cell_width):
        cdef _box.Box cBox = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new locality.LinkCell(cBox, float(cell_width))
        self._nlist = NeighborList()

    def __dealloc__(self):
        del self.thisptr

    def getBox(self):
        """
        :return: Freud Box
        :rtype: :py:class:`freud.box.Box`
        """
        return BoxFromCPP(self.thisptr.getBox())

    def getNumCells(self):
        """
        :return: the number of cells in this box
        :rtype: unsigned int
        """
        return self.thisptr.getNumCells()

    def getCell(self, point):
        """Returns the index of the cell containing the given point

        :param point: point coordinates :math:`\\left(x,y,z\\right)`
        :type point: :class:`numpy.ndarray`, shape= :math:`\\left(3\\right)`, dtype= :class:`numpy.float32`
        :return: cell index
        :rtype: unsigned int
        """
        point = freud.common.convert_array(point, 1, dtype=np.float32, contiguous=True,
            dim_message="point must be a 1 dimensional array")

        cdef float[:] cPoint = point

        return self.thisptr.getCell(dereference(<vec3[float]*>&cPoint[0]))

    def itercell(self, unsigned int cell):
        """Return an iterator over all particles in the given cell

        :param cell: Cell index
        :type cell: unsigned int
        :return: iterator to particle indices in specified cell
        :rtype: iter
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
        :type cell: unsigned int
        :return: array of cell neighbors
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{neighbors}\\right)`, dtype= :class:`numpy.uint32`
        """
        neighbors = self.thisptr.getCellNeighbors(int(cell))
        result = np.zeros(neighbors.size(), dtype=np.uint32)
        for i in range(neighbors.size()):
            result[i] = neighbors[i]
        return result

    def computeCellList(self, box, ref_points, points=None, exclude_ii=None):
        """Update the data structure for the given set of points and compute a
        NeighborList

        :param box: simulation box
        :param ref_points: reference point coordinates
        :param points: point coordinates
        :param exlude_ii: True if pairs of points with identical indices should be excluded; if None, is set to True if points is None or the same object as ref_points
        :type box: :py:class:`freud.box.Box`
        :type ref_points: :class:`numpy.ndarray`, shape= :math:`\\left(N_{refpoints}, 3\\right)`, dtype= :class:`numpy.float32`
        :type points: :class:`numpy.ndarray`, shape= :math:`\\left(N_{points}, 3\\right)`, dtype= :class:`numpy.float32`
        """
        exclude_ii = (points is ref_points or points is None) if exclude_ii is None else exclude_ii

        ref_points = freud.common.convert_array(ref_points, 2, dtype=np.float32, contiguous=True,
            dim_message="ref_points must be a 2 dimensional array")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        if points is None:
            points = ref_points

        points = freud.common.convert_array(points, 2, dtype=np.float32, contiguous=True,
            dim_message="points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')
        cdef _box.Box cBox = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
            box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        cdef np.ndarray cRefPoints = ref_points
        cdef unsigned int Nref = ref_points.shape[0]
        cdef np.ndarray cPoints = points
        cdef unsigned int Np = points.shape[0]
        cdef cbool c_exclude_ii = exclude_ii
        with nogil:
            self.thisptr.compute(cBox, <vec3[float]*> cRefPoints.data, Nref, <vec3[float]*> cPoints.data, Np, c_exclude_ii)

        cdef locality.NeighborList *nlist = self.thisptr.getNeighborList()
        self._nlist.refer_to(nlist)
        return self

    def compute(self, box, ref_points, points=None, exclude_ii=None):
        """Update the data structure for the given set of points and compute a
        NeighborList

        :param box: simulation box
        :param ref_points: reference point coordinates
        :param points: point coordinates
        :param exlude_ii: True if pairs of points with identical indices should be excluded; if None, is set to True if points is None or the same object as ref_points
        :type box: :py:class:`freud.box.Box`
        :type ref_points: :class:`numpy.ndarray`, shape= :math:`\\left(N_{refpoints}, 3\\right)`, dtype= :class:`numpy.float32`
        :type points: :class:`numpy.ndarray`, shape= :math:`\\left(N_{points}, 3\\right)`, dtype= :class:`numpy.float32`
        """
        return self.computeCellList(box, ref_points, points, exclude_ii)

    @property
    def nlist(self):
        """Returns the neighbor list stored by this object, generated by
        `compute`."""
        return self._nlist

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
    :type n_neigh: unsigned int
    :type scale: float
    :type strict_cut: bool

    Example::

       nn = NearestNeighbors(2, 6)
       nn.compute(box, positions, positions)
       hexatic = order.HexOrderParameter(2)
       hexatic.compute(box, positions, nlist=nn.nlist)
    """
    cdef locality.NearestNeighbors *thisptr
    cdef NeighborList _nlist
    cdef _cached_points
    cdef _cached_ref_points
    cdef _cached_box

    def __cinit__(self, float rmax, unsigned int n_neigh, float scale=1.1, strict_cut=False):
        if scale < 1:
            raise RuntimeError("scale must be greater than 1")
        self.thisptr = new locality.NearestNeighbors(float(rmax), int(n_neigh), float(scale), bool(strict_cut))
        self._nlist = NeighborList()

    def __dealloc__(self):
        del self.thisptr

    def getUINTMAX(self):
        """
        :return: value of C++ UINTMAX used to pad the arrays
        :rtype: unsigned int
        """
        return self.thisptr.getUINTMAX()

    def getBox(self):
        """
        :return: Freud Box
        :rtype: :py:class:`freud.box.Box`
        """
        return BoxFromCPP(self.thisptr.getBox())

    def getNumNeighbors(self):
        """
        :return: the number of neighbors this object will find
        :rtype: unsigned int
        """
        return self.thisptr.getNumNeighbors()

    def getNRef(self):
        """
        :return: the number of particles this object found neighbors of
        :rtype: unsigned int
        """
        return self.thisptr.getNref()

    def setRMax(self, float rmax):
        """Update the neighbor search distance guess
        :param rmax: nearest neighbors search radius
        :type rmax: float
        """
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
        """Return the current neighbor search distance guess
        :return: nearest neighbors search radius
        :rtype: float
        """
        return self.thisptr.getRMax()

    def getNeighbors(self, unsigned int i):
        """Return the N nearest neighbors of the reference point with index i

        :param i: index of the reference point to fetch the neighboring points of
        :type i: unsigned int
        """
        cdef unsigned int nNeigh = self.thisptr.getNumNeighbors()
        result = np.empty(nNeigh, dtype=np.uint32)
        result[:] = self.getUINTMAX()
        cdef unsigned int start_idx = self.nlist.find_first_index(i)
        cdef unsigned int end_idx = self.nlist.find_first_index(i + 1)
        result[:end_idx - start_idx] = self.nlist.index_j[start_idx:end_idx]

        return result

    def getNeighborList(self):
        """Return the entire neighbors list

        :return: Neighbor List
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}, N_{neighbors}\\right)`, dtype= :class:`numpy.uint32`
        """
        result = np.empty((self.thisptr.getNref(), self.thisptr.getNumNeighbors()), dtype=np.uint32)
        result[:] = self.getUINTMAX()
        idx_i, idx_j = self.nlist.index_i, self.nlist.index_j
        cdef size_t num_bonds = len(self.nlist.index_i)
        cdef size_t last_i = 0
        cdef size_t current_j = 0
        for bond in range(num_bonds):
            current_j *= last_i == idx_i[bond]
            last_i = idx_i[bond]
            result[last_i, current_j] = idx_j[bond]
            current_j += 1

        return result

    def getRsq(self, unsigned int i):
        """
        Return the Rsq values for the N nearest neighbors of the reference point with index i

        :param i: index of the reference point of which to fetch the neighboring point distances
        :type i: unsigned int
        :return: squared distances of the N nearest neighbors
        :return: Neighbor List
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}\\right)`, dtype= :class:`numpy.float32`
        """
        cdef unsigned int start_idx = self.nlist.find_first_index(i)
        cdef unsigned int end_idx = self.nlist.find_first_index(i + 1)
        rijs = (self._cached_points[self.nlist.index_j[start_idx:end_idx]] -
                self._cached_ref_points[self.nlist.index_i[start_idx:end_idx]])
        self._cached_box.wrap(rijs)
        result = -np.ones((self.thisptr.getNumNeighbors(),), dtype=np.float32)
        result[:len(rijs)] = np.sum(rijs**2, axis=-1)
        return result

    def getWrappedVectors(self):
        """
        Return the wrapped vectors for computed neighbors. Array padded with -1 for empty neighbors

        :return: wrapped vectors
        :return: Neighbor List
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}\\right)`, dtype= :class:`numpy.float32`
        """
        return self._getWrappedVectors()[0]

    def _getWrappedVectors(self):
        result = np.empty((self.thisptr.getNref(), self.thisptr.getNumNeighbors(), 3), dtype=np.float32)
        blank_mask = np.ones((self.thisptr.getNref(), self.thisptr.getNumNeighbors()), dtype=np.bool)
        idx_i, idx_j = self.nlist.index_i, self.nlist.index_j
        cdef size_t num_bonds = len(self.nlist.index_i)
        cdef size_t last_i = 0
        cdef size_t current_j = 0
        for bond in range(num_bonds):
            current_j *= last_i == idx_i[bond]
            last_i = idx_i[bond]
            result[last_i, current_j] = self._cached_points[idx_j[bond]] - self._cached_ref_points[last_i]
            blank_mask[last_i, current_j] = False
            current_j += 1

        self._cached_box.wrap(result.reshape((-1, 3)))
        result[blank_mask] = -1
        return result, blank_mask

    def getRsqList(self):
        """
        Return the entire Rsq values list

        :return: Rsq list
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}, N_{neighbors}\\right)`, dtype= :class:`numpy.float32`
        """
        (vecs, blank_mask) = self._getWrappedVectors()
        result = np.sum(vecs**2, axis=-1)
        result[blank_mask] = -1
        return result

    def compute(self, box, ref_points, points, exclude_ii=None):
        """Update the data structure for the given set of points

        :param box: simulation box
        :param ref_points: coordinated of reference points
        :param points: coordinates of points
        :param exlude_ii: True if pairs of points with identical indices should be excluded; if None, is set to True if points is None or the same object as ref_points
        :type box: :py:class:`freud.box.Box`
        :type ref_points: :class:`numpy.ndarray`, shape=(:math:`N_{particles}`, 3), dtype= :class:`numpy.float32`
        :type points: :class:`numpy.ndarray`, shape=(:math:`N_{particles}`, 3), dtype= :class:`numpy.float32`
        """
        exclude_ii = (points is ref_points or points is None) if exclude_ii is None else exclude_ii

        ref_points = freud.common.convert_array(ref_points, 2, dtype=np.float32, contiguous=True,
            dim_message="ref_points must be a 2 dimensional array")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        points = freud.common.convert_array(points, 2, dtype=np.float32, contiguous=True,
            dim_message="points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        self._cached_ref_points = ref_points
        self._cached_points = points
        self._cached_box = box

        cdef _box.Box cBox = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        cdef np.ndarray cRef_points = ref_points
        cdef unsigned int n_ref = ref_points.shape[0]
        cdef np.ndarray cPoints = points
        cdef unsigned int Np = points.shape[0]
        cdef cbool c_exclude_ii = exclude_ii
        with nogil:
            self.thisptr.compute(cBox, <vec3[float]*> cRef_points.data, n_ref, <vec3[float]*> cPoints.data, Np, c_exclude_ii)

        cdef locality.NeighborList *nlist = self.thisptr.getNeighborList()
        self._nlist.refer_to(nlist)

        return self

    @property
    def nlist(self):
        """Returns the neighbor list stored by this object, generated by
        `compute`."""
        return self._nlist
