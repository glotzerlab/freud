# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The locality module contains data structures to efficiently locate points based
on their proximity to other points.
"""

import sys
import numpy as np
import freud.common

from libcpp cimport bool as cbool
from freud.util._VectorMath cimport vec3
from cython.operator cimport dereference
cimport freud._locality
cimport freud.box

cimport numpy as np

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


cdef class NeighborList:
    """Class representing a certain number of "bonds" between
    particles. Computation methods will iterate over these bonds when
    searching for neighboring particles.

    NeighborList objects are constructed for two sets of position
    arrays A (alternatively *reference points*; of length :math:`n_A`)
    and B (alternatively *target points*; of length :math:`n_B`) and
    hold a set of :math:`\\left(i, j\\right): i < n_A, j < n_B` index
    pairs corresponding to near-neighbor points in A and B,
    respectively.

    For efficiency, all bonds for a particular reference particle :math:`i`
    are contiguous and bonds are stored in order based on reference
    particle index :math:`i`. The first bond index corresponding to a given
    particle can be found in :math:`\\log(n_{bonds})` time using
    :py:meth:`find_first_index`.

    .. moduleauthor:: Matthew Spellings <mspells@umich.edu>

    .. versionadded:: 0.6.4

    .. note::

       Typically, there is no need to instantiate this class directly.
       In most cases, users should manipulate
       :py:class:`freud.locality.NeighborList` objects received from a
       neighbor search algorithm, such as :py:class:`freud.locality.LinkCell`,
       :py:class:`freud.locality.NearestNeighbors`, or
       :py:class:`freud.voronoi.Voronoi`.

    Attributes:
        index_i (:class:`np.ndarray`):
            The reference point indices from the last set of points this object
            was evaluated with. This array is read-only to prevent breakage of
            :py:meth:`~.find_first_index()`.
        index_j (:class:`np.ndarray`):
            The reference point indices from the last set of points this object
            was evaluated with. This array is read-only to prevent breakage of
            :py:meth:`~.find_first_index()`.
        weights ((:math:`N_{bonds}`) :class:`np.ndarray`):
            The per-bond weights from the last set of points this object was
            evaluated with.
        segments ((:math:`N_{ref\_points}`) :class:`np.ndarray`):
            A segment array, which is an array of length :math:`N_{ref}`
            indicating the first bond index for each reference particle from
            the last set of points this object was evaluated with.
        neighbor_counts ((:math:`N_{ref\_points}`) :class:`np.ndarray`):
            A neighbor count array, which is an array of length
            :math:`N_{ref}` indicating the number of neighbors for each
            reference particle from the last set of points this object was
            evaluated with.

    Example::

       # Assume we have position as Nx3 array
       lc = LinkCell(box, 1.5).compute(box, positions)
       nlist = lc.nlist

       # Get all vectors from central particles to their neighbors
       rijs = positions[nlist.index_j] - positions[nlist.index_i]
       box.wrap(rijs)
    """ # noqa

    @classmethod
    def from_arrays(cls, Nref, Ntarget, index_i, index_j, weights=None):
        """Create a NeighborList from a set of bond information arrays.

        Args:
            Nref (int):
                Number of reference points (corresponding to :code:`index_i`).
            Ntarget (int):
                Number of target points (corresponding to :code:`index_j`).
            index_i (np.array):
                Array of integers corresponding to indices in the set of
                reference points.
            index_j (np.array):
                Array of integers corresponding to indices in the set of
                target points.
            weights (np.array, optional):
                Array of per-bond weights (if :code:`None` is given, use a
                value of 1 for each weight) (Default value = :code:`None`).
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
        cdef size_t i
        if n_bonds:
            last_i = c_index_i[0]
            i = last_i
            for bond in range(n_bonds):
                i = c_index_i[bond]
                if i < last_i:
                    raise RuntimeError('index_i is not sorted')
                if c_Nref <= i:
                    raise RuntimeError(
                        'Nref is too small for a value found in index_i')
                if c_Ntarget <= c_index_j[bond]:
                    raise RuntimeError(
                        'Ntarget is too small for a value found in index_j')
                last_i = i

        result = cls()
        cdef NeighborList c_result = result
        c_result.thisptr.resize(n_bonds)
        cdef size_t * c_neighbors_ptr = c_result.thisptr.getNeighbors()
        cdef float * c_weights_ptr = c_result.thisptr.getWeights()

        for bond in range(n_bonds):
            c_neighbors_ptr[2*bond] = c_index_i[bond]
            c_neighbors_ptr[2*bond + 1] = c_index_j[bond]
            c_weights_ptr[bond] = c_weights[bond]

        c_result.thisptr.setNumBonds(n_bonds, c_Nref, c_Ntarget)

        return result

    cdef refer_to(self, freud._locality.NeighborList * other):
        """Makes this cython wrapper object point to a different C++ object,
        deleting the one we are already holding if necessary. We do not
        own the memory of the other C++ object."""
        if self._managed:
            del self.thisptr
        self._managed = False
        self.thisptr = other

    def __cinit__(self):
        self._managed = True
        self.thisptr = new freud._locality.NeighborList()

    def __dealloc__(self):
        if self._managed:
            del self.thisptr

    cdef freud._locality.NeighborList * get_ptr(self) nogil:
        """Returns a pointer to the raw C++ object we are wrapping."""
        return self.thisptr

    cdef void copy_c(self, NeighborList other):
        """Copies the contents of other into this object."""
        self.thisptr.copy(dereference(other.thisptr))

    def copy(self, other=None):
        """Create a copy. If other is given, copy its contents into this
        object. Otherwise, return a copy of this object.

        Args:
            other (:py:class:`freud.locality.NeighborList`, optional):
                A NeighborList to copy into this object (Default value =
                :code:`None`).
        """
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
        cdef np.npy_intp size[2]
        size[0] = self.thisptr.getNumBonds()
        size[1] = 2
        self.thisptr.getNeighbors()
        cdef np.ndarray[np.uint64_t, ndim=2] result = \
            np.PyArray_SimpleNewFromData(2, size, np.NPY_UINT64,
                                         <void*> self.thisptr.getNeighbors())
        result.flags.writeable = False
        return result[:, 0]

    @property
    def index_j(self):
        cdef np.npy_intp size[2]
        size[0] = self.thisptr.getNumBonds()
        size[1] = 2
        cdef np.ndarray[np.uint64_t, ndim=2] result = \
            np.PyArray_SimpleNewFromData(2, size, np.NPY_UINT64,
                                         <void*> self.thisptr.getNeighbors())
        result.flags.writeable = False
        return result[:, 1]

    @property
    def weights(self):
        cdef np.npy_intp size[1]
        size[0] = self.thisptr.getNumBonds()
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, size, np.NPY_FLOAT32,
                                         <void*> self.thisptr.getWeights())
        return result

    @property
    def segments(self):
        result = np.zeros((self.thisptr.getNumI(),), dtype=np.int64)
        cdef size_t * neighbors = self.thisptr.getNeighbors()
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
        cdef np.ndarray[np.int64_t, ndim=1] result = np.zeros(
            (self.thisptr.getNumI(),), dtype=np.int64)
        cdef size_t * neighbors = self.thisptr.getNeighbors()
        cdef int last_i = -1
        cdef int i = -1
        cdef size_t n = 0
        for bond in range(self.thisptr.getNumBonds()):
            i = neighbors[2*bond]
            if i != last_i and i > 0:
                result[last_i] = n
                n = 0
            last_i = i
            n += 1

        if last_i >= 0:
            result[last_i] = n

        return result

    def __len__(self):
        """Returns the number of bonds stored in this object."""
        return self.thisptr.getNumBonds()

    def find_first_index(self, unsigned int i):
        """Returns the lowest bond index corresponding to a reference particle
        with an index :math:`\\geq i`.

        Args:
            i (unsigned int ): The particle index.
        """
        return self.thisptr.find_first_index(i)

    def filter(self, filt):
        """Removes bonds that satisfy a boolean criterion.

        Args:
            filt (:class:`np.ndarray`):
                Boolean-like array of bonds to keep (True means the bond
                will not be removed).

        .. note:: This method modifies this object in-place.

        Example::

            # Keep only the bonds between particles of type A and type B
            nlist.filter(types[nlist.index_i] != types[nlist.index_j])
        """
        filt = np.ascontiguousarray(filt, dtype=np.bool)
        cdef np.ndarray[np.uint8_t, ndim=1, cast=True] filt_c = filt
        cdef cbool * filt_ptr = <cbool*> filt_c.data
        self.thisptr.filter(filt_ptr)
        return self

    def filter_r(self, box, ref_points, points, float rmax, float rmin=0):
        """Removes bonds that are outside of a given radius range.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points to use for filtering.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Target points to use for filtering.
            rmax (float):
                Maximum bond distance in the resulting neighbor list.
            rmin (float, optional):
                Minimum bond distance in the resulting neighbor list
                (Default value = 0).
        """
        cdef freud.box.Box b = freud.common.convert_box(box)
        ref_points = freud.common.convert_array(
            ref_points, 2, dtype=np.float32, contiguous=True,
            array_name="ref_points")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        cdef np.ndarray cRef_points = ref_points
        cdef np.ndarray cPoints = points
        cdef size_t nRef = ref_points.shape[0]
        cdef size_t nP = points.shape[0]

        self.thisptr.validate(nRef, nP)
        self.thisptr.filter_r(
            dereference(b.thisptr),
            <vec3[float]*> cRef_points.data,
            <vec3[float]*> cPoints.data,
            rmax,
            rmin)
        return self


def make_default_nlist(box, ref_points, points, rmax, nlist=None,
                       exclude_ii=None):
    """Helper function to return a neighbor list object if is given, or to
    construct one using LinkCell if it is not.

    Args:
        box (:class:`freud.box.Box`):
            Simulation box.
        ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
            Reference points for the neighborlist.
        points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
            Points to construct the neighborlist.
        rmax (float):
            The radius within which to find neighbors.
        nlist (:class:`freud.locality.NeighborList`, optional):
            NeighborList to use to find bonds (Default value = :code:`None`).
        exclude_ii (bool, optional):
            True if pairs of points with identical indices should be excluded;
            if None, is set to True if points is None or the same object as
            ref_points (Default value = :code:`None`).

    Returns:
        tuple (:class:`freud.locality.NeighborList`, \
        :class:`freud.locality:LinkCell`):
            The neighborlist and the owning LinkCell object.
    """
    if nlist is not None:
        return nlist, nlist
    cdef freud.box.Box b = freud.common.convert_box(box)

    cdef LinkCell lc = LinkCell(box, rmax).computeCellList(
        box, ref_points, points, exclude_ii)

    # Python does not appear to garbage collect appropriately in this case.
    # If a new neighbor list is created, the associated link cell keeps the
    # reference to it alive even if it goes out of scope in the calling
    # program, and since the neighbor list also references the link cell the
    # resulting cycle causes a memory leak. The below block explicitly breaks
    # this cycle. Alternatively, we could force garbage collection using the
    # gc module, but this is simpler.
    cdef NeighborList cnlist = lc.nlist
    if nlist is None:
        cnlist.base = None

    # Return the owner of the neighbor list as well to prevent gc problems
    return lc.nlist, lc


def make_default_nlist_nn(box, ref_points, points, n_neigh, nlist=None,
                          exclude_ii=None, rmax_guess=2.0):
    """Helper function to return a neighbor list object if is given, or to
    construct one using NearestNeighbors if it is not.

    Args:
        box (:class:`freud.box.Box`):
            Simulation box.
        ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
            Reference points for the neighborlist.
        points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
            Points to construct the neighborlist.
        n_neigh (int):
            The number of nearest neighbors to consider.
        nlist (:class:`freud.locality.NeighborList`, optional):
            NeighborList to use to find bonds (Default value = :code:`None`).
        exclude_ii (bool, optional):
            True if pairs of points with identical indices should be excluded;
            if None, is set to True if points is None or the same object as
            ref_points (Default value = :code:`None`).
        rmax_guess (float):
            Estimate of rmax, speeds up search if chosen properly.

    Returns:
        tuple (:class:`freud.locality.NeighborList`, \
        :class:`freud.locality:NearestNeighbors`):
            The neighborlist and the owning NearestNeighbors object.
    """
    if nlist is not None:
        return nlist, nlist
    cdef freud.box.Box b = freud.common.convert_box(box)

    cdef NearestNeighbors nn = NearestNeighbors(rmax_guess, n_neigh).compute(
        box, ref_points, points)

    # Python does not appear to garbage collect appropriately in this case.
    # If a new neighbor list is created, the associated link cell keeps the
    # reference to it alive even if it goes out of scope in the calling
    # program, and since the neighbor list also references the link cell the
    # resulting cycle causes a memory leak. The below block explicitly breaks
    # this cycle. Alternatively, we could force garbage collection using the
    # gc module, but this is simpler.
    cdef NeighborList cnlist = nn.nlist
    if nlist is None:
        cnlist.base = None

    # Return the owner of the neighbor list as well to prevent gc problems
    return nn.nlist, nn


cdef class IteratorLinkCell:
    """Iterates over the particles in a cell.

    .. moduleauthor:: Joshua Anderson <joaander@umich.edu>

    Example::

       # Grab particles in cell 0
       for j in linkcell.itercell(0):
           print(positions[j])
    """
    def __cinit__(self):
        # Must be running python 3.x
        current_version = sys.version_info
        if current_version.major < 3:
            raise RuntimeError(
                "Must use python 3.x or greater to use IteratorLinkCell")
        else:
            self.thisptr = new freud._locality.IteratorLinkCell()

    def __dealloc__(self):
        del self.thisptr

    cdef void copy(self, const freud._locality.IteratorLinkCell & rhs):
        self.thisptr.copy(rhs)

    def next(self):
        """Implements iterator interface"""
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

    Args:
        box (:py:class:`freud.box.Box`):
            Simulation box.
        cell_width (float):
            Maximum distance to find particles within.

    Attributes:
        box (:class:`freud.box.Box`):
            Simulation box.
        num_cells (unsigned int):
            The number of cells in the box.
        nlist (:class:`freud.locality.NeighborList`):
            The neighbor list stored by this object, generated by
            :py:meth:`~.compute()`.
        index_i (:class:`np.ndarray`):
            The reference point indices from the last set of points this object
            was evaluated with. This array is read-only to prevent breakage of
            :py:meth:`~.find_first_index()`.
        index_j (:class:`np.ndarray`):
            The reference point indices from the last set of points this object
            was evaluated with. This array is read-only to prevent breakage of
            :py:meth:`~.find_first_index()`.
        weights ((:math:`N_{bonds}`) :class:`np.ndarray`):
            The per-bond weights from the last set of points this object was
            evaluated with.

    .. note::
        2D: :py:class:`freud.locality.LinkCell` properly handles 2D boxes.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    Example::

       # Assume positions are an Nx3 array
       lc = LinkCell(box, 1.5)
       lc.computeCellList(box, positions)
       for i in range(positions.shape[0]):
           # Cell containing particle i
           cell = lc.getCell(positions[0])
           # List of cell's neighboring cells
           cellNeighbors = lc.getCellNeighbors(cell)
           # Iterate over neighboring cells (including our own)
           for neighborCell in cellNeighbors:
               # Iterate over particles in each neighboring cell
               for neighbor in lc.itercell(neighborCell):
                   pass # Do something with neighbor index

       # Using NeighborList API
       dens = density.LocalDensity(1.5, 1, 1)
       dens.compute(box, positions, nlist=lc.nlist)
    """
    def __cinit__(self, box, cell_width):
        cdef freud.box.Box b = freud.common.convert_box(box)
        self.thisptr = new freud._locality.LinkCell(
            dereference(b.thisptr), float(cell_width))
        self._nlist = NeighborList()

    def __dealloc__(self):
        del self.thisptr

    @property
    def box(self):
        return self.getBox()

    def getBox(self):
        """Get the freud Box.

        Returns:
            :class:`freud.box.Box`: freud Box.
        """
        return freud.box.BoxFromCPP(self.thisptr.getBox())

    @property
    def num_cells(self):
        return self.getNumCells()

    def getNumCells(self):
        """Get the number of cells in this box.

        Returns:
            unsigned int: The number of cells in this box.
        """
        return self.thisptr.getNumCells()

    def getCell(self, point):
        """Returns the index of the cell containing the given point.

        Args:
            point(:math:`\\left(3\\right)` :class:`numpy.ndarray`):
                Point coordinates :math:`\\left(x,y,z\\right)`.

        Returns:
            unsigned int: Cell index.
        """
        point = freud.common.convert_array(
            point, 1, dtype=np.float32, contiguous=True, array_name="point")

        cdef float[:] cPoint = point

        return self.thisptr.getCell(dereference(<vec3[float]*> &cPoint[0]))

    def itercell(self, unsigned int cell):
        """Return an iterator over all particles in the given cell.

        Args:
            cell (unsigned int): Cell index.

        Returns:
            iter: Iterator to particle indices in specified cell.
        """
        current_version = sys.version_info
        if current_version.major < 3:
            raise RuntimeError(
                "Must use python 3.x or greater to use itercell")
        result = IteratorLinkCell()
        cdef freud._locality.IteratorLinkCell cResult = self.thisptr.itercell(
            cell)
        result.copy(cResult)
        return iter(result)

    def getCellNeighbors(self, cell):
        """Returns the neighboring cell indices of the given cell.

        Args:
            cell (unsigned int): Cell index.

        Returns:
            :math:`\\left(N_{neighbors}\\right)` :class:`numpy.ndarray`:
                Array of cell neighbors.
        """
        neighbors = self.thisptr.getCellNeighbors(int(cell))
        result = np.zeros(neighbors.size(), dtype=np.uint32)
        for i in range(neighbors.size()):
            result[i] = neighbors[i]
        return result

    def computeCellList(self, box, ref_points, points=None, exclude_ii=None):
        """Update the data structure for the given set of points and compute a
        NeighborList.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference point coordinates.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`, \
            optional):
                Point coordinates (Default value = :code:`None`).
            exclude_ii (bool, optional):
                True if pairs of points with identical indices should be
                excluded; if None, is set to True if points is None or the same
                object as ref_points (Default value = :code:`None`).
        """
        cdef freud.box.Box b = freud.common.convert_box(box)
        exclude_ii = (
            points is ref_points or points is None) \
            if exclude_ii is None else exclude_ii

        ref_points = freud.common.convert_array(
            ref_points, 2, dtype=np.float32, contiguous=True,
            array_name="ref_points")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        if points is None:
            points = ref_points

        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        cdef np.ndarray cRef_points = ref_points
        cdef unsigned int n_ref = ref_points.shape[0]
        cdef np.ndarray cPoints = points
        cdef unsigned int Np = points.shape[0]
        cdef cbool c_exclude_ii = exclude_ii
        with nogil:
            self.thisptr.compute(
                dereference(b.thisptr),
                <vec3[float]*> cRef_points.data,
                n_ref,
                <vec3[float]*> cPoints.data,
                Np,
                c_exclude_ii)

        cdef freud._locality.NeighborList * nlist
        nlist = self.thisptr.getNeighborList()
        self._nlist.refer_to(nlist)
        self._nlist.base = self
        return self

    def compute(self, box, ref_points, points=None, exclude_ii=None):
        """Update the data structure for the given set of points and compute a
        NeighborList.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference point coordinates.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`, \
            optional):
                Point coordinates (Default value = :code:`None`).
            exclude_ii (bool, optional):
                True if pairs of points with identical indices should be
                excluded; if None, is set to True if points is None or the same
                object as ref_points (Default value = :code:`None`).
        """
        return self.computeCellList(box, ref_points, points, exclude_ii)

    @property
    def nlist(self):
        return self._nlist

cdef class NearestNeighbors:
    """Supports efficiently finding the :math:`N` nearest neighbors of each
    point in a set for some fixed integer :math:`N`.

    * :code:`strict_cut == True`: :code:`rmax` will be strictly obeyed, and any
      particle which has fewer than :math:`N` neighbors will have values of
      :code:`UINT_MAX` assigned.
    * :code:`strict_cut == False` (default): :code:`rmax` will be expanded to
      find the requested number of neighbors. If :code:`rmax` increases to the
      point that a cell list cannot be constructed, a warning will be raised
      and the neighbors already found will be returned.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    Args:
        rmax (float):
            Initial guess of a distance to search within to find N neighbors.
        n_neigh (unsigned int):
            Number of neighbors to find for each point.
        scale (float):
            Multiplier by which to automatically increase :code:`rmax` value if
            the requested number of neighbors is not found. Only utilized if
            :code:`strict_cut` is False. Scale must be greater than 1.
        strict_cut (bool):
            Whether to use a strict :code:`rmax` or allow for automatic
            expansion, default is False.

    Attributes:
        UINTMAX (unsigned int):
            Value of C++ UINTMAX used to pad the arrays.
        box (:class:`freud.box.Box`):
            Simulation box.
        num_neighbors (unsigned int):
            The number of neighbors this object will find.
        n_ref (unsigned int):
            The number of particles this object found neighbors of.
        r_max (float):
            Current nearest neighbors search radius guess.
        wrapped_vectors (:math:`\\left(N_{particles}\\right)` \
        :class:`numpy.ndarray`):
            The wrapped vectors padded with -1 for empty neighbors.
        r_sq_list (:math:`\\left(N_{particles}, N_{neighbors}\\right)` \
        :class:`numpy.ndarray`):
            The Rsq values list.
        nlist (:class:`freud.locality.NeighborList`):
            The neighbor list stored by this object, generated by
            :py:meth:`~.compute()`.

    Example::

       nn = NearestNeighbors(2, 6)
       nn.compute(box, positions, positions)
       hexatic = order.HexOrderParameter(2)
       hexatic.compute(box, positions, nlist=nn.nlist)
    """
    def __cinit__(self, float rmax, unsigned int n_neigh, float scale=1.1,
                  strict_cut=False):
        if scale < 1:
            raise RuntimeError("scale must be greater than 1")
        self.thisptr = new freud._locality.NearestNeighbors(
            float(rmax), int(n_neigh), float(scale), bool(strict_cut))
        self._nlist = NeighborList()

    def __dealloc__(self):
        del self.thisptr

    @property
    def UINTMAX(self):
        return self.getUINTMAX()

    def getUINTMAX(self):
        """
        Returns:
            unsigned int: Value of C++ UINTMAX used to pad the arrays.
        """
        return self.thisptr.getUINTMAX()

    @property
    def box(self):
        return self.getBox()

    def getBox(self):
        """Get the freud Box.

        Returns:
            :class:`freud.box.Box`: freud Box.
        """
        return freud.box.BoxFromCPP(self.thisptr.getBox())

    @property
    def num_neighbors(self):
        return self.getNumNeighbors()

    def getNumNeighbors(self):
        """The number of neighbors this object will find.

        Returns:
            unsigned int: The number of neighbors this object will find.
        """
        return self.thisptr.getNumNeighbors()

    @property
    def n_ref(self):
        return self.getNRef()

    def getNRef(self):
        """Get the number of particles this object found neighbors of.

        Returns:
            unsigned int:
                The number of particles this object found neighbors of.
        """
        return self.thisptr.getNref()

    def setRMax(self, float rmax):
        warnings.warn("Use constructor arguments instead of this setter. "
                      "This setter will be removed in the future.",
                      FreudDeprecationWarning)
        self.thisptr.setRMax(rmax)

    def setCutMode(self, strict_cut):
        """Set mode to handle :code:`rmax` by Nearest Neighbors.

        * :code:`strict_cut == True`: :code:`rmax` will be strictly obeyed,
          and any particle which has fewer than :math:`N` neighbors will have
          values of :code:`UINT_MAX` assigned.
        * :code:`strict_cut == False`: :code:`rmax` will be expanded to find
          the requested number of neighbors. If :code:`rmax` increases to the
          point that a cell list cannot be constructed, a warning will be
          raised and the neighbors already found will be returned.

        Args:
            strict_cut (bool): Whether to use a strict :code:`rmax` or allow
                for automatic expansion.
        """
        warnings.warn("Use constructor arguments instead of this setter. "
                      "This setter will be removed in the future.",
                      FreudDeprecationWarning)
        self.thisptr.setCutMode(strict_cut)

    @property
    def r_max(self):
        return self.getRMax()

    def getRMax(self):
        """Return the current neighbor search distance guess.

        Returns:
            float: Nearest neighbors search radius.
        """
        return self.thisptr.getRMax()

    def getNeighbors(self, unsigned int i):
        """Return the :math:`N` nearest neighbors of the reference point with
        index :math:`i`.

        Args:
            i (unsigned int):
                Index of the reference point whose neighbors will be returned.
        """
        cdef unsigned int nNeigh = self.thisptr.getNumNeighbors()
        result = np.empty(nNeigh, dtype=np.uint32)
        result[:] = self.getUINTMAX()
        cdef unsigned int start_idx = self.nlist.find_first_index(i)
        cdef unsigned int end_idx = self.nlist.find_first_index(i + 1)
        result[:end_idx - start_idx] = self.nlist.index_j[start_idx:end_idx]

        return result

    def getNeighborList(self):
        """Return the entire neighbor list.

        Returns:
            :math:`\\left(N_{particles}, N_{neighbors}\\right)` \
            :class:`numpy.ndarray`:
                Neighbor List.
        """
        result = np.empty(
            (self.thisptr.getNref(), self.thisptr.getNumNeighbors()),
            dtype=np.uint32)
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
        """Return the squared distances to the :math:`N` nearest neighbors of
        the reference point with index :math:`i`.

        Args:
            i (unsigned int):
                Index of the reference point of which to fetch the neighboring
                point distances.

        Returns:
            :math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`:
                Squared distances to the :math:`N` nearest neighbors.
        """
        cdef unsigned int start_idx = self.nlist.find_first_index(i)
        cdef unsigned int end_idx = self.nlist.find_first_index(i + 1)
        rijs = (self._cached_points[self.nlist.index_j[start_idx:end_idx]] -
                self._cached_ref_points[self.nlist.index_i[start_idx:end_idx]])
        self._cached_box.wrap(rijs)
        result = -np.ones((self.thisptr.getNumNeighbors(),), dtype=np.float32)
        result[:len(rijs)] = np.sum(rijs**2, axis=-1)
        return result

    @property
    def wrapped_vectors(self):
        return self.getWrappedVectors()

    def getWrappedVectors(self):
        """Return the wrapped vectors for computed neighbors. Array padded
        with -1 for empty neighbors.

        Returns:
            :math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`:
                Wrapped vectors.
        """
        return self._getWrappedVectors()[0]

    def _getWrappedVectors(self):
        result = np.empty(
            (self.thisptr.getNref(), self.thisptr.getNumNeighbors(), 3),
            dtype=np.float32)
        blank_mask = np.ones(
            (self.thisptr.getNref(), self.thisptr.getNumNeighbors()),
            dtype=np.bool)
        idx_i, idx_j = self.nlist.index_i, self.nlist.index_j
        cdef size_t num_bonds = len(self.nlist.index_i)
        cdef size_t last_i = 0
        cdef size_t current_j = 0
        for bond in range(num_bonds):
            current_j *= last_i == idx_i[bond]
            last_i = idx_i[bond]
            result[last_i, current_j] = \
                self._cached_points[idx_j[bond]] - \
                self._cached_ref_points[last_i]
            blank_mask[last_i, current_j] = False
            current_j += 1

        self._cached_box.wrap(result.reshape((-1, 3)))
        result[blank_mask] = -1
        return result, blank_mask

    @property
    def r_sq_list(self):
        return self.getRsqList()

    def getRsqList(self):
        """Return the entire Rsq values list.

        Returns:
            :math:`\\left(N_{particles}, N_{neighbors}\\right)` \
            :class:`numpy.ndarray`:
                Rsq list.
        """
        (vecs, blank_mask) = self._getWrappedVectors()
        result = np.sum(vecs**2, axis=-1)
        result[blank_mask] = -1
        return result

    def compute(self, box, ref_points, points=None, exclude_ii=None):
        """Update the data structure for the given set of points.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference point coordinates.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`, \
            optional):
                Point coordinates. Defaults to :code:`ref_points` if not
                provided or :code:`None`.
            exclude_ii (bool, optional):
                True if pairs of points with identical indices should be
                excluded; if None, is set to True if points is None or the same
                object as ref_points (Default value = :code:`None`).
        """
        cdef freud.box.Box b = freud.common.convert_box(box)
        exclude_ii = (
            points is ref_points or points is None) \
            if exclude_ii is None else exclude_ii

        ref_points = freud.common.convert_array(
            ref_points, 2, dtype=np.float32, contiguous=True,
            array_name="ref_points")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        if points is None:
            points = ref_points

        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        self._cached_ref_points = ref_points
        self._cached_points = points
        self._cached_box = box

        cdef np.ndarray cRef_points = ref_points
        cdef unsigned int n_ref = ref_points.shape[0]
        cdef np.ndarray cPoints = points
        cdef unsigned int Np = points.shape[0]
        cdef cbool c_exclude_ii = exclude_ii
        with nogil:
            self.thisptr.compute(
                dereference(b.thisptr),
                <vec3[float]*> cRef_points.data,
                n_ref,
                <vec3[float]*> cPoints.data,
                Np,
                c_exclude_ii)

        cdef freud._locality.NeighborList * nlist
        nlist = self.thisptr.getNeighborList()
        self._nlist.refer_to(nlist)
        self._nlist.base = self
        return self

    @property
    def nlist(self):
        return self._nlist
