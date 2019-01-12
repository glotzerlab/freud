# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The :class:`freud.locality` module contains data structures to efficiently
locate points based on their proximity to other points.
"""

import sys
import numpy as np
import freud.common
import warnings

from freud.errors import FreudDeprecationWarning

from libcpp cimport bool as cbool
from freud.util._VectorMath cimport vec3
from cython.operator cimport dereference
from cython.operator cimport dereference
from libcpp.memory cimport shared_ptr
from freud._locality cimport ITERATOR_TERMINATOR

cimport freud._locality
cimport freud.box
cimport numpy as np

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


cdef class NeighborQueryResult:
    R"""Class encapsulating the output of queries of NeighborQuery objects.

    The NeighborQueryResult makes it easy to work with the results of queries
    and convert them to various natural objects. Additionally, the result is a
    generator, making it easy for users to lazily iterate over the object. This
    class should never be instantiated directly, but rather used as the output
    of the query or queryBall functions of a NeighborQuery instance.

    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    .. versionadded:: 0.12.0
    """
    def __iter__(self):
        cdef freud._locality.NeighborPoint npoint
        cdef vec3[float] l_cur_point
        cdef unsigned int i

        if self.query_type == 'nn':
            for i in range(self.Np):
                l_cur_point = vec3[float](self.points[i, 0], self.points[i, 1], self.points[i, 2])
                self.iterator = self.spdptr.query(l_cur_point, self.k)

                while True:
                    npoint = dereference(self.iterator).next()
                    if npoint == ITERATOR_TERMINATOR:
                        break
                    elif self.exclude_ii and npoint.id == i:
                        continue
                    yield (i, npoint.id, npoint.distance)
        else:
            for i in range(self.Np):
                l_cur_point = vec3[float](self.points[i, 0], self.points[i, 1], self.points[i, 2])
                self.iterator = self.spdptr.queryBall(l_cur_point, self.r)

                while True:
                    npoint = dereference(self.iterator).next()
                    if npoint == ITERATOR_TERMINATOR:
                        break
                    elif self.exclude_ii and npoint.id == i:
                        continue
                    yield (i, npoint.id, npoint.distance)

        raise StopIteration

    def toList(self):
        """Convert query result to a list."""
        neighbors = []
        for neigh in self:
            neighbors.append(neigh)

        return neighbors

    def toNList(self):
        """Convert query result to a freud NeighborList."""
        index_i = []
        index_j = []
        for neigh in self:
            index_i.append(neigh[0])
            index_j.append(neigh[1])

        index_j = np.array(index_j)
        Ntarget = np.unique(index_j).shape[0]
        return NeighborList.from_arrays(
            self.Np, Ntarget, np.asarray(index_i), index_j)


cdef class AABBQueryResult(NeighborQueryResult):
    R"""Extend NeighborQuery class to call the correct iterator query function.

    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    .. versionadded:: 0.12.0
    """
    def __iter__(self):
        cdef freud._locality.NeighborPoint npoint
        cdef vec3[float] l_cur_point
        cdef unsigned int i

        for i in range(self.Np):
            l_cur_point = vec3[float](self.points[i, 0], self.points[i, 1], self.points[i, 2])
            self.iterator = self.aabbptr.query(
                l_cur_point, self.k, self.r_guess, self.scale)

            while True:
                npoint = dereference(self.iterator).next()
                if npoint == ITERATOR_TERMINATOR:
                    break
                elif self.exclude_ii and npoint.id == i:
                    continue
                yield (i, npoint.id, npoint.distance)

        raise StopIteration


cdef class NeighborQuery:
    R"""Class representing a set of points along with the ability to query for
    neighbors of these points.

    The NeighborQuery class represents the abstract interface for neighbor
    finding. The class contains a set of points and a simulation box, the
    latter of which is used to define the system and the periodic boundary
    conditions required for finding neighbors of these points. The primary mode
    of interacting with the NeighborQuery is through the query and queryBall
    functions, which enable finding either the nearest neighbors of a point or
    all points within a distance cutoff, respectively. Subclasses of
    NeighborQuery implement these methods based on the nature of the underlying
    data structure.

    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    .. versionadded:: 0.12.0

    Args:
        box (:class:`freud.box.Box`):
            Simulation box.
        points ((:math:`N`, 3) :class:`numpy.ndarray`):
            Point coordinates to build the structure.

    Attributes:
        box (:class:`freud.box.Box`):
            The box object used by this data structure.
        ref_points (:class:`np.ndarray`):
            The array of points in this data structure.
    """

    def __cinit__(self):
        if type(self) is NeighborQuery:
            raise RuntimeError(
                "The NeighborQuery class is abstract, and should not be "
                "directly instantiated"
            )

    def __dealloc__(self):
        pass

    @property
    def box(self):
        return self.box

    @property
    def ref_points(self):
        return np.asarray(self.ref_points)

    def query(self, points, unsigned int k=1, cbool exclude_ii=False):
        R"""Query the tree for nearest neighbors of the provided point.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N`, 3) :class:`numpy.ndarray`):
                Points to query for.
            k (int):
                The number of nearest neighbors to find.

        Returns:
            (:math:`N`, :math:`k`) :class:`numpy.ndarray`:
                Array of indices of the :math:`k` nearest neighbors for each
                input point.
        """
        # Can't use this function with old-style NeighborQuery objects
        if not self.queryable:
            raise RuntimeError("You cannot use the query method unless this "
                               "object was originally constructed with "
                               "reference points")
        points = freud.common.convert_array(
            np.atleast_2d(points), 2, dtype=np.float32, contiguous=True,
            array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        # Ensure that enough neighbors are found when excluding
        if exclude_ii:
            k += 1

        return NeighborQueryResult.init(self.spdptr, points, exclude_ii, r=0, k=k)

    def queryBall(self, points, float r, cbool exclude_ii=False):
        R"""Query the tree for all points within a distance r of the provided point(s).

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N`, 3) :class:`numpy.ndarray`):
                Points to query for.
            r (float):
                The distance within which to find neighbors

        Returns:
            list or list[list]:
                If the input was a single point, returns a list of
                its neighbors. Otherwise, returns a list of lists
                of the neighbors.
        """
        if not self.queryable:
            raise RuntimeError("You cannot use the query method unless this "
                               "object was originally constructed with "
                               "reference points")
        points = freud.common.convert_array(
            np.atleast_2d(points), 2, dtype=np.float32, contiguous=True,
            array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        return NeighborQueryResult.init(self.spdptr, points, exclude_ii, r=r, k=0)


cdef class NeighborList:
    R"""Class representing a certain number of "bonds" between
    particles. Computation methods will iterate over these bonds when
    searching for neighboring particles.

    NeighborList objects are constructed for two sets of position
    arrays A (alternatively *reference points*; of length :math:`n_A`)
    and B (alternatively *target points*; of length :math:`n_B`) and
    hold a set of :math:`\left(i, j\right): i < n_A, j < n_B` index
    pairs corresponding to near-neighbor points in A and B,
    respectively.

    For efficiency, all bonds for a particular reference particle :math:`i`
    are contiguous and bonds are stored in order based on reference
    particle index :math:`i`. The first bond index corresponding to a given
    particle can be found in :math:`\log(n_{bonds})` time using
    :meth:`find_first_index`.

    .. moduleauthor:: Matthew Spellings <mspells@umich.edu>

    .. versionadded:: 0.6.4

    .. note::

       Typically, there is no need to instantiate this class directly.
       In most cases, users should manipulate
       :class:`freud.locality.NeighborList` objects received from a
       neighbor search algorithm, such as :class:`freud.locality.LinkCell`,
       :class:`freud.locality.NearestNeighbors`, or
       :class:`freud.voronoi.Voronoi`.

    Attributes:
        index_i (:class:`np.ndarray`):
            The reference point indices from the last set of points this object
            was evaluated with. This array is read-only to prevent breakage of
            :meth:`~.find_first_index()`.
        index_j (:class:`np.ndarray`):
            The reference point indices from the last set of points this object
            was evaluated with. This array is read-only to prevent breakage of
            :meth:`~.find_first_index()`.
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
    """

    @classmethod
    def from_arrays(cls, Nref, Ntarget, index_i, index_j, weights=None):
        R"""Create a NeighborList from a set of bond information arrays.

        Args:
            Nref (int):
                Number of reference points (corresponding to :code:`index_i`).
            Ntarget (int):
                Number of target points (corresponding to :code:`index_j`).
            index_i (:class:`np.ndarray`):
                Array of integers corresponding to indices in the set of
                reference points.
            index_j (:class:`np.ndarray`):
                Array of integers corresponding to indices in the set of
                target points.
            weights (:class:`np.ndarray`, optional):
                Array of per-bond weights (if :code:`None` is given, use a
                value of 1 for each weight) (Default value = :code:`None`).
        """
        index_i = freud.common.convert_array(index_i, dimensions=1,
                                             dtype=np.uint64, contiguous=True,
                                             array_name='index_i')
        index_j = freud.common.convert_array(index_j, dimensions=1,
                                             dtype=np.uint64, contiguous=True,
                                             array_name='index_j')

        if index_i.shape != index_j.shape:
            raise TypeError('index_i and index_j should be the same size')

        if weights is None:
            weights = np.ones(index_i.shape, dtype=np.float32)
        else:
            weights = freud.common.convert_array(
                weights, dimensions=1, dtype=np.float32, contiguous=True,
                array_name='weights')

        if weights.shape != index_i.shape:
            raise TypeError('weights and index_i should be the same size')

        cdef size_t[::1] c_index_i = index_i
        cdef size_t[::1] c_index_j = index_j
        cdef float[::1] c_weights = weights
        cdef size_t n_bonds = c_index_i.shape[0]
        cdef size_t c_Nref = Nref
        cdef size_t c_Ntarget = Ntarget

        cdef size_t bond
        cdef size_t last_i
        cdef size_t i
        if n_bonds > 0:
            last_i = c_index_i[0]
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
        R"""Makes this cython wrapper object point to a different C++ object,
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
        R"""Returns a pointer to the raw C++ object we are wrapping."""
        return self.thisptr

    cdef void copy_c(self, NeighborList other):
        R"""Copies the contents of other into this object."""
        self.thisptr.copy(dereference(other.thisptr))

    def copy(self, other=None):
        R"""Create a copy. If other is given, copy its contents into this
        object. Otherwise, return a copy of this object.

        Args:
            other (:class:`freud.locality.NeighborList`, optional):
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
        cdef size_t n_bonds = self.thisptr.getNumBonds()
        cdef size_t[:, ::1] neighbors
        if not n_bonds:
            result = np.asarray([], dtype=np.uint64)
        else:
            neighbors = <size_t[:n_bonds, :2]> self.thisptr.getNeighbors()
            result = np.asarray(neighbors[:, 0], dtype=np.uint64)
        result.flags.writeable = False
        return result

    @property
    def index_j(self):
        cdef size_t n_bonds = self.thisptr.getNumBonds()
        cdef size_t[:, ::1] neighbors
        if not n_bonds:
            result = np.asarray([], dtype=np.uint64)
        else:
            neighbors = <size_t[:n_bonds, :2]> self.thisptr.getNeighbors()
            result = np.asarray(neighbors[:, 1], dtype=np.uint64)
        result.flags.writeable = False
        return result

    @property
    def weights(self):
        cdef size_t n_bonds = self.thisptr.getNumBonds()
        if not n_bonds:
            return np.asarray([], dtype=np.float32)
        cdef float[::1] weights = \
            <float[:n_bonds]> self.thisptr.getWeights()
        return np.asarray(weights)

    @property
    def segments(self):
        cdef np.ndarray[np.int64_t, ndim=1] result = np.zeros(
            (self.thisptr.getNumI(),), dtype=np.int64)
        cdef size_t n_bonds = self.thisptr.getNumBonds()
        if not n_bonds:
            return result
        cdef size_t[:, ::1] neighbors = \
            <size_t[:n_bonds, :2]> self.thisptr.getNeighbors()
        cdef int last_i = -1
        cdef int i = -1
        cdef size_t bond
        for bond in range(n_bonds):
            i = neighbors[bond, 0]
            if i != last_i:
                result[i] = bond
            last_i = i
        return result

    @property
    def neighbor_counts(self):
        cdef np.ndarray[np.int64_t, ndim=1] result = np.zeros(
            (self.thisptr.getNumI(),), dtype=np.int64)
        cdef size_t n_bonds = self.thisptr.getNumBonds()
        if not n_bonds:
            return result
        cdef size_t[:, ::1] neighbors = \
            <size_t[:n_bonds, :2]> self.thisptr.getNeighbors()
        cdef int last_i = -1
        cdef int i = -1
        cdef size_t n = 0
        cdef size_t bond
        for bond in range(n_bonds):
            i = neighbors[bond, 0]
            if i != last_i and i > 0:
                if last_i >= 0:
                    result[last_i] = n
                n = 0
            last_i = i
            n += 1

        if last_i >= 0:
            result[last_i] = n

        return result

    def __len__(self):
        R"""Returns the number of bonds stored in this object."""
        return self.thisptr.getNumBonds()

    def find_first_index(self, unsigned int i):
        R"""Returns the lowest bond index corresponding to a reference particle
        with an index :math:`\geq i`.

        Args:
            i (unsigned int ): The particle index.
        """
        return self.thisptr.find_first_index(i)

    def filter(self, filt):
        R"""Removes bonds that satisfy a boolean criterion.

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
        R"""Removes bonds that are outside of a given radius range.

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

        cdef float[:, ::1] cRef_points = ref_points
        cdef float[:, ::1] cPoints = points
        cdef size_t nRef = ref_points.shape[0]
        cdef size_t nP = points.shape[0]

        self.thisptr.validate(nRef, nP)
        self.thisptr.filter_r(
            dereference(b.thisptr),
            <vec3[float]*> &cRef_points[0, 0],
            <vec3[float]*> &cPoints[0, 0],
            rmax,
            rmin)
        return self


def make_default_nlist(box, ref_points, points, rmax, nlist=None,
                       exclude_ii=None):
    R"""Helper function to return a neighbor list object if is given, or to
    construct one using AABBQuery if it is not.

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
            Set this to :code:`True` if pairs of points with identical
            indices should be excluded. If this is :code:`None`, it will be
            treated as :code:`True` if :code:`points` is :code:`None` or
            the same object as :code:`ref_points` (Defaults to
            :code:`None`).

    Returns:
        tuple (:class:`freud.locality.NeighborList`, :class:`freud.locality.AABBQuery`):
            The neighborlist and the owning AABBQuery object.
    """  # noqa: E501
    if nlist is not None:
        return nlist, nlist

    cdef AABBQuery aq = AABBQuery().compute(
        box, rmax, ref_points, points, exclude_ii)

    # Python does not appear to garbage collect appropriately in this case.
    # If a new neighbor list is created, the associated owner keeps the
    # reference to it alive even if it goes out of scope in the calling
    # program, and since the neighbor list also references the link cell the
    # resulting cycle causes a memory leak. The below block explicitly breaks
    # this cycle. Alternatively, we could force garbage collection using the
    # gc module, but this is simpler.
    cdef NeighborList cnlist = aq.nlist
    if nlist is None:
        cnlist.base = None

    # Return the owner of the neighbor list as well to prevent gc problems
    return aq.nlist, aq


def make_default_nlist_nn(box, ref_points, points, n_neigh, nlist=None,
                          exclude_ii=None, rmax_guess=2.0):
    R"""Helper function to return a neighbor list object if is given, or to
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
            Set this to :code:`True` if pairs of points with identical
            indices should be excluded. If this is :code:`None`, it will be
            treated as :code:`True` if :code:`points` is :code:`None` or
            the same object as :code:`ref_points` (Defaults to
            :code:`None`).
        rmax_guess (float):
            Estimate of rmax, speeds up search if chosen properly.

    Returns:
        tuple (:class:`freud.locality.NeighborList`, :class:`freud.locality:NearestNeighbors`):
            The neighborlist and the owning NearestNeighbors object.
    """  # noqa: E501
    if nlist is not None:
        return nlist, nlist

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


cdef class AABBQuery(NeighborQuery):
    R"""Use an AABB tree to find neighbors.

    .. moduleauthor:: Bradley Dice <bdice@bradleydice.com>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Attributes:
        box (:class:`freud.locality.Box`):
            The simulation box.
        ref_points (:class:`np.ndarray`):
            The points associated with this class.
    """  # noqa: E501

    def __cinit__(self, box, ref_points):
        cdef float[:, ::1] l_ref_points
        if type(self) is AABBQuery:
            # Assume valid set of arguments is passed
            self.queryable = True
            self.box = freud.common.convert_box(box)
            self.ref_points = freud.common.convert_array(
                ref_points.copy(), 2, dtype=np.float32, contiguous=True,
                array_name="ref_points")
            l_ref_points = self.ref_points
            self.thisptr = self.spdptr = new freud._locality.AABBQuery(
                dereference(self.box.thisptr),
                <vec3[float]*> &l_ref_points[0, 0],
                self.ref_points.shape[0])

    def __dealloc__(self):
        if type(self) is AABBQuery:
            del self.thisptr

    def query(self, points, unsigned int k=1, float r=0, float scale=1.1,
              cbool exclude_ii=False):
        R"""Query the tree for nearest neighbors of the provided point.

        The AABBQuery object overrides the parent method to support querying
        based on a specified guessed rcut and scaling.
        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N`, 3) :class:`numpy.ndarray`):
                Points to query for.
            k (int):
                The number of nearest neighbors to find.
            r (float):
                The initial guess of a distance to search to find N neighbors.
            scale (float):
                Multiplier by which to increase :code:`r` if not enough
                neighbors are found.

        Returns:
            (:math:`N`, :math:`k`) :class:`numpy.ndarray`:
                Array of indices of the :math:`k` nearest neighbors for each
                input point.
        """
        points = freud.common.convert_array(
            np.atleast_2d(points), 2, dtype=np.float32, contiguous=True,
            array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        # Ensure that enough neighbors are found when excluding
        if exclude_ii:
            k += 1

        # Default guess value
        if r == 0:
            r = 0.1*min(self.box.L)

        return AABBQueryResult.init2(self.thisptr, points, exclude_ii, k, r, scale)


cdef class IteratorLinkCell:
    R"""Iterates over the particles in a cell.

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
        R"""Implements iterator interface"""
        cdef unsigned int result = self.thisptr.next()
        if self.thisptr.atEnd():
            raise StopIteration()
        return result

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


cdef class LinkCell(NeighborQuery):
    R"""Supports efficiently finding all points in a set within a certain
    distance from a given point.

    .. moduleauthor:: Joshua Anderson <joaander@umich.edu>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        box (:class:`freud.box.Box`):
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
            :meth:`~.compute()`.

    .. note::
        **2D:** :class:`freud.locality.LinkCell` properly handles 2D boxes.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    Example::

       # Assume positions are an Nx3 array
       lc = LinkCell(box, 1.5)
       lc.compute(box, positions)
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

    def __cinit__(self, box, cell_width, ref_points=None):
        self.box = freud.common.convert_box(box)
        cdef float[:, ::1] l_ref_points
        if ref_points is not None:
            # The new API
            self.queryable = True
            self.ref_points = freud.common.convert_array(
                ref_points.copy(), 2, dtype=np.float32, contiguous=True,
                array_name="ref_points")
            l_ref_points = self.ref_points
            self.thisptr = self.spdptr = new freud._locality.LinkCell(
                dereference(self.box.thisptr), float(cell_width),
                <vec3[float]*> &l_ref_points[0, 0],
                self.ref_points.shape[0])
        else:
            # The old API
            self.queryable = False
            self.thisptr = self.spdptr = new freud._locality.LinkCell(
                dereference(self.box.thisptr), float(cell_width))
        self._nlist = NeighborList()

    def __dealloc__(self):
        del self.thisptr

    def getBox(self):
        warnings.warn("The getBox function is deprecated in favor "
                      "of the box class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.box

    @property
    def num_cells(self):
        return self.thisptr.getNumCells()

    def getNumCells(self):
        warnings.warn("The getNumCells function is deprecated in favor "
                      "of the num_cells class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.num_cells

    def getCell(self, point):
        R"""Returns the index of the cell containing the given point.

        Args:
            point(:math:`\left(3\right)` :class:`numpy.ndarray`):
                Point coordinates :math:`\left(x,y,z\right)`.

        Returns:
            unsigned int: Cell index.
        """
        point = freud.common.convert_array(
            point, 1, dtype=np.float32, contiguous=True, array_name="point")

        cdef float[::1] cPoint = point

        return self.thisptr.getCell(dereference(<vec3[float]*> &cPoint[0]))

    def itercell(self, unsigned int cell):
        R"""Return an iterator over all particles in the given cell.

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
        R"""Returns the neighboring cell indices of the given cell.

        Args:
            cell (unsigned int): Cell index.

        Returns:
            :math:`\left(N_{neighbors}\right)` :class:`numpy.ndarray`:
                Array of cell neighbors.
        """
        neighbors = self.thisptr.getCellNeighbors(int(cell))
        result = np.zeros(neighbors.size(), dtype=np.uint32)
        for i in range(neighbors.size()):
            result[i] = neighbors[i]
        return result

    def compute(self, box, ref_points, points=None, exclude_ii=None):
        R"""Update the data structure for the given set of points and compute a
        NeighborList.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference point coordinates.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`, optional):
                Point coordinates (Default value = :code:`None`).
            exclude_ii (bool, optional):
                Set this to :code:`True` if pairs of points with identical
                indices should be excluded. If this is :code:`None`, it will be
                treated as :code:`True` if :code:`points` is :code:`None` or
                the same object as :code:`ref_points` (Defaults to
                :code:`None`).
        """  # noqa: E501
        if self.queryable:
            raise RuntimeError("You cannot use the compute method because "
                               "this object was originally constructed with "
                               "reference points")
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

        cdef float[:, ::1] cRef_points = ref_points
        cdef unsigned int n_ref = ref_points.shape[0]
        cdef float[:, ::1] cPoints = points
        cdef unsigned int Np = points.shape[0]
        cdef cbool c_exclude_ii = exclude_ii
        with nogil:
            self.thisptr.compute(
                dereference(b.thisptr),
                <vec3[float]*> &cRef_points[0, 0],
                n_ref,
                <vec3[float]*> &cPoints[0, 0],
                Np,
                c_exclude_ii)

        cdef freud._locality.NeighborList * nlist
        nlist = self.thisptr.getNeighborList()
        self._nlist.refer_to(nlist)
        self._nlist.base = self
        return self

    def computeCellList(self, box, ref_points, points=None, exclude_ii=None):
        warnings.warn("The computeCellList function is deprecated in favor "
                      "of the compute method and will be removed in a future "
                      "version of freud.",
                      FreudDeprecationWarning)
        return self.compute(box, ref_points, points, exclude_ii)

    @property
    def nlist(self):
        return self._nlist


cdef class NearestNeighbors:
    R"""Supports efficiently finding the :math:`N` nearest neighbors of each
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
        wrapped_vectors (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The wrapped vectors padded with -1 for empty neighbors.
        r_sq_list (:math:`\left(N_{particles}, N_{neighbors}\right)` :class:`numpy.ndarray`):
            The Rsq values list.
        nlist (:class:`freud.locality.NeighborList`):
            The neighbor list stored by this object, generated by
            :meth:`~.compute()`.

    Example::

       nn = NearestNeighbors(2, 6)
       nn.compute(box, positions, positions)
       hexatic = order.HexOrderParameter(2)
       hexatic.compute(box, positions, nlist=nn.nlist)
    """  # noqa: E501

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
        return self.thisptr.getUINTMAX()

    def getUINTMAX(self):
        warnings.warn("The getUINTMAX function is deprecated in favor "
                      "of the UINTMAX class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.UINTMAX

    @property
    def box(self):
        return freud.box.BoxFromCPP(self.thisptr.getBox())

    def getBox(self):
        warnings.warn("The getBox function is deprecated in favor "
                      "of the box class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.box

    @property
    def num_neighbors(self):
        return self.thisptr.getNumNeighbors()

    def getNumNeighbors(self):
        warnings.warn("The getNumNeighbors function is deprecated in favor "
                      "of the num_neighbors class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.num_neighbors

    @property
    def n_ref(self):
        return self.thisptr.getNref()

    def getNRef(self):
        warnings.warn("The getNref function is deprecated in favor "
                      "of the n_ref class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.n_ref

    def setRMax(self, float rmax):
        warnings.warn("Use constructor arguments instead of this setter. "
                      "This setter will be removed in the future.",
                      FreudDeprecationWarning)
        self.thisptr.setRMax(rmax)

    def setCutMode(self, strict_cut):
        R"""Set mode to handle :code:`rmax` by Nearest Neighbors.

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
        return self.thisptr.getRMax()

    def getRMax(self):
        warnings.warn("The getRMax function is deprecated in favor "
                      "of the r_max class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.r_max

    def getNeighbors(self, unsigned int i):
        R"""Return the :math:`N` nearest neighbors of the reference point with
        index :math:`i`.

        Args:
            i (unsigned int):
                Index of the reference point whose neighbors will be returned.
        Returns:
            :math:`\left(N_{neighbors}\right)` :class:`numpy.ndarray`:
                Indices of points that are neighbors of reference point
                :math:`i`, padded with UINTMAX if fewer neighbors than
                requested were found.
        """
        cdef unsigned int nNeigh = self.thisptr.getNumNeighbors()
        result = np.empty(nNeigh, dtype=np.uint32)
        result[:] = self.UINTMAX
        cdef unsigned int start_idx = self.nlist.find_first_index(i)
        cdef unsigned int end_idx = self.nlist.find_first_index(i + 1)
        result[:end_idx - start_idx] = self.nlist.index_j[start_idx:end_idx]

        return result

    def getNeighborList(self):
        R"""Return the entire neighbor list.

        Returns:
            :math:`\left(N_{particles}, N_{neighbors}\right)` :class:`numpy.ndarray`:
                Indices of up to :math:`N_{neighbors}` points that are
                neighbors of the :math:`N_{particles}` reference points, padded
                with UINTMAX if fewer neighbors than requested were found.
        """  # noqa: E501
        result = np.empty(
            (self.thisptr.getNref(), self.thisptr.getNumNeighbors()),
            dtype=np.uint32)
        result[:] = self.UINTMAX
        idx_i, idx_j = self.nlist.index_i, self.nlist.index_j
        cdef size_t num_bonds = len(self.nlist.index_i)
        cdef size_t bond
        cdef size_t last_i = 0
        cdef size_t current_j = 0
        for bond in range(num_bonds):
            current_j *= last_i == idx_i[bond]
            last_i = idx_i[bond]
            result[last_i, current_j] = idx_j[bond]
            current_j += 1

        return result

    def getRsq(self, unsigned int i):
        R"""Return the squared distances to the :math:`N` nearest neighbors of
        the reference point with index :math:`i`.

        Args:
            i (unsigned int):
                Index of the reference point of which to fetch the neighboring
                point distances.

        Returns:
            :math:`\left(N_{particles}\right)` :class:`numpy.ndarray`:
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
        return self._getWrappedVectors()[0]

    def getWrappedVectors(self):
        warnings.warn("The getWrappedVectors function is deprecated in favor "
                      "of the wrapped_vectors class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.wrapped_vectors

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
        (vecs, blank_mask) = self._getWrappedVectors()
        result = np.sum(vecs**2, axis=-1)
        result[blank_mask] = -1
        return result

    def getRsqList(self):
        warnings.warn("The getRsqList function is deprecated in favor "
                      "of the r_sq_list class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.r_sq_list

    def compute(self, box, ref_points, points=None, exclude_ii=None):
        R"""Update the data structure for the given set of points.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference point coordinates.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`, optional):
                Point coordinates. Defaults to :code:`ref_points` if not
                provided or :code:`None`.
            exclude_ii (bool, optional):
                Set this to :code:`True` if pairs of points with identical
                indices should be excluded. If this is :code:`None`, it will be
                treated as :code:`True` if :code:`points` is :code:`None` or
                the same object as :code:`ref_points` (Defaults to
                :code:`None`).
        """  # noqa: E501
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
        self._cached_box = b

        cdef float[:, ::1] cRef_points = ref_points
        cdef unsigned int n_ref = ref_points.shape[0]
        cdef float[:, ::1] cPoints = points
        cdef unsigned int Np = points.shape[0]
        cdef cbool c_exclude_ii = exclude_ii
        with nogil:
            self.thisptr.compute(
                dereference(b.thisptr),
                <vec3[float]*> &cRef_points[0, 0],
                n_ref,
                <vec3[float]*> &cPoints[0, 0],
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
