# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The :mod:`freud.locality` module contains data structures to efficiently
locate points based on their proximity to other points.
"""
import sys
import numpy as np
import freud.common
import itertools
import warnings
import logging
import copy

from libcpp cimport bool as cbool
from freud.util cimport vec3
from cython.operator cimport dereference
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from freud._locality cimport ITERATOR_TERMINATOR
from freud.common cimport Compute

cimport freud._locality
cimport freud.box
cimport numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.spatial import Voronoi as qvoronoi
    from scipy.spatial import ConvexHull
    _SCIPY_AVAILABLE = True
except ImportError:
    qvoronoi = None
    msg = ('scipy.spatial.Voronoi is not available (requires scipy 0.12+), '
           'so freud.voronoi is not available.')
    logger.warning(msg)
    _SCIPY_AVAILABLE = False


# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class _QueryArgs:
    R"""Container for query arguments.

    This class is use internally throughout freud to provide a nice interface
    between keyword- or dict-style query arguments and the C++ QueryArgs
    object. All arguments are funneled through this interface, which constructs
    the appropriate C++ QueryArgs object that can then be used in C++ compute
    calls.

    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>
    """

    def __cinit__(self, mode=None, r_min=None, r_max=None, r_guess=None,
                  num_neighbors=None, exclude_ii=None,
                  scale=None, **kwargs):
        if type(self) == _QueryArgs:
            self.thisptr = new freud._locality.QueryArgs()
            self.mode = mode
            if r_max is not None:
                self.r_max = r_max
            if r_min is not None:
                self.r_min = r_min
            if r_guess is not None:
                self.r_guess = r_guess
            if num_neighbors is not None:
                self.num_neighbors = num_neighbors
            if exclude_ii is not None:
                self.exclude_ii = exclude_ii
            if scale is not None:
                self.scale = scale
            if len(kwargs):
                err_str = ", ".join(
                    "{} = {}".format(k, v) for k, v in kwargs.items())
                raise ValueError(
                    "The following invalid query "
                    "arguments were provided: " +
                    err_str)

    def __dealloc__(self):
        if type(self) == _QueryArgs:
            del self.thisptr

    def update(self, qargs):
        if qargs is None:
            return
        for arg in qargs:
            if hasattr(self, arg):
                setattr(self, arg, qargs[arg])
            else:
                raise ValueError("You have passed an invalid query argument")

    @classmethod
    def from_dict(cls, mapping):
        """Create _QueryArgs from mapping."""
        return cls(**mapping)

    @property
    def mode(self):
        if self.thisptr.mode == freud._locality.QueryType.none:
            return None
        elif self.thisptr.mode == freud._locality.QueryType.ball:
            return 'ball'
        elif self.thisptr.mode == freud._locality.QueryType.nearest:
            return 'nearest'
        else:
            raise ValueError("Unknown mode {} set!".format(self.thisptr.mode))

    @mode.setter
    def mode(self, value):
        if value == 'none' or value is None:
            self.thisptr.mode = freud._locality.QueryType.none
        elif value == 'ball':
            self.thisptr.mode = freud._locality.QueryType.ball
        elif value == 'nearest':
            self.thisptr.mode = freud._locality.QueryType.nearest
        else:
            raise ValueError("You have passed an invalid mode.")

    @property
    def r_guess(self):
        return self.thisptr.r_guess

    @r_guess.setter
    def r_guess(self, value):
        self.thisptr.r_guess = value

    @property
    def r_min(self):
        return self.thisptr.r_min

    @r_min.setter
    def r_min(self, value):
        self.thisptr.r_min = value

    @property
    def r_max(self):
        return self.thisptr.r_max

    @r_max.setter
    def r_max(self, value):
        self.thisptr.r_max = value

    @property
    def num_neighbors(self):
        return self.thisptr.num_neighbors

    @num_neighbors.setter
    def num_neighbors(self, value):
        self.thisptr.num_neighbors = value

    @property
    def exclude_ii(self):
        return self.thisptr.exclude_ii

    @exclude_ii.setter
    def exclude_ii(self, value):
        self.thisptr.exclude_ii = value

    @property
    def scale(self):
        return self.thisptr.scale

    @scale.setter
    def scale(self, value):
        self.thisptr.scale = value


cdef class NeighborQueryResult:
    R"""Class encapsulating the output of queries of NeighborQuery objects.

    .. warning::

        This class should not be instantiated directly, it is the
        return value of all `query*` functions of
        :class:`~NeighborQuery`. The class provides a convenient
        interface for iterating over query results, and can be
        transparently converted into a list or a
        :class:`~NeighborList` object.

    The :class:`~NeighborQueryResult` makes it easy to work with the results of
    queries and convert them to various natural objects. Additionally, the
    result is a generator, making it easy for users to lazily iterate over the
    object.

    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>
    """

    def __iter__(self):
        cdef freud._locality.NeighborBond npoint

        cdef const float[:, ::1] l_points = self.points
        cdef shared_ptr[freud._locality.NeighborQueryIterator] iterator = \
            self.nq.nqptr.query(
                <vec3[float]*> &l_points[0, 0],
                self.points.shape[0],
                dereference(self.query_args.thisptr))

        npoint = dereference(iterator).next()
        while npoint != ITERATOR_TERMINATOR:
            yield (npoint.id, npoint.ref_id, npoint.distance)
            npoint = dereference(iterator).next()

        raise StopIteration

    def toNeighborList(self):
        """Convert query result to a freud NeighborList.

        Returns:
            :class:`~NeighborList`: A :mod:`freud` :class:`~NeighborList`
            containing all neighbor pairs found by the query generating this
            result object.
        """
        cdef const float[:, ::1] l_points = self.points
        cdef shared_ptr[freud._locality.NeighborQueryIterator] iterator = \
            self.nq.nqptr.query(
                <vec3[float]*> &l_points[0, 0],
                self.points.shape[0],
                dereference(self.query_args.thisptr))

        cdef freud._locality.NeighborList *cnlist = dereference(
            iterator).toNeighborList()
        cdef NeighborList nl = NeighborList()
        nl.refer_to(cnlist)
        # Explicitly manage a manually created nlist so that it will be
        # deleted when the Python object is.
        nl._managed = True

        return nl


cdef class NeighborQuery:
    R"""Class representing a set of points along with the ability to query for
    neighbors of these points.

    .. warning::

        This class should not be instantiated directly. The subclasses
        :class:`~AABBQuery` and :class:`~LinkCell` provide the
        intended interfaces.

    The :class:`~.NeighborQuery` class represents the abstract interface for
    neighbor finding. The class contains a set of points and a simulation box,
    the latter of which is used to define the system and the periodic boundary
    conditions required for finding neighbors of these points. The primary mode
    of interacting with the :class:`~.NeighborQuery` is through the
    :meth:`~NeighborQuery.query` and :meth:`~NeighborQuery.queryBall`
    functions, which enable finding either the nearest neighbors of a point or
    all points within a distance cutoff, respectively.  Subclasses of
    NeighborQuery implement these methods based on the nature of the underlying
    data structure.

    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        box (:class:`freud.box.Box`):
            Simulation box.
        points ((:math:`N`, 3) :class:`numpy.ndarray`):
            Point coordinates to build the structure.

    Attributes:
        box (:class:`freud.box.Box`):
            The box object used by this data structure.
        points (:class:`np.ndarray`):
            The array of points in this data structure.
    """

    def __cinit__(self):
        if type(self) is NeighborQuery:
            raise RuntimeError(
                "The NeighborQuery class is abstract, and should not be "
                "directly instantiated"
            )

    @property
    def box(self):
        return self._box

    @property
    def points(self):
        return np.asarray(self.points)

    def query(self, query_points, query_args):
        R"""Query for nearest neighbors of the provided point.

        Args:
            query_points ((:math:`N`, 3) :class:`numpy.ndarray`):
                Points to query for.
            query_args (dict):
                Query arguments determining how to find neighbors. For
                information on valid query argument, see the documentation of
                `~._QueryArgs`.

        Returns:
            :class:`~.NeighborQueryResult`: Results object containing the
            output of this query.
        """
        query_points = freud.common.convert_array(
            np.atleast_2d(query_points), shape=(None, 3))

        cdef _QueryArgs args = _QueryArgs.from_dict(query_args)
        return NeighborQueryResult.init(self, query_points, args)

    cdef freud._locality.NeighborQuery * get_ptr(self):
        R"""Returns a pointer to the raw C++ object we are wrapping."""
        return self.nqptr


cdef class NeighborList:
    R"""Class representing bonds between two sets of points.

    Compute classes contain a set of bonds between two sets of position
    arrays ("query points" and "points") and hold a list of index pairs
    :math:`\left(i, j\right)` where
    :math:`i < N_{query\_points}, j < N_{points}` corresponding to neighbor
    pairs between the two sets.

    For efficiency, all bonds must be sorted by the query point index, from
    least to greatest. Bonds have an query point index :math:`i` and a point
    index :math:`j`. The first bond index corresponding to a given query point
    can be found in :math:`\log(N_{bonds})` time using
    :meth:`find_first_index`, because bonds are ordered by the query point
    index.

    .. moduleauthor:: Matthew Spellings <mspells@umich.edu>
    .. moduleauthor:: Bradley Dice <bdice@bradleydice.com>

    .. note::

       Typically, there is no need to instantiate this class directly.
       In most cases, users should manipulate
       :class:`freud.locality.NeighborList` objects received from a
       neighbor search algorithm, such as :class:`freud.locality.LinkCell`,
       :class:`freud.locality.AABBQuery`, or :class:`freud.locality.Voronoi`.

    Attributes:
        query_point_indices ((:math:`N_{bonds}`) :class:`np.ndarray`):
            The query point indices for each bond. This array is read-only to
            prevent breakage of :meth:`~.find_first_index()`. Equivalent to
            indexing with :code:`[:, 0]`.
        point_indices ((:math:`N_{bonds}`) :class:`np.ndarray`):
            The point indices for each bond. This array is read-only to
            prevent breakage of :meth:`~.find_first_index()`. Equivalent to
            indexing with :code:`[:, 1]`.
        weights ((:math:`N_{bonds}`) :class:`np.ndarray`):
            The weights for each bond. By default, bonds have a weight of 1.
        distances ((:math:`N_{bonds}`) :class:`np.ndarray`):
            The distances for each bond.
        segments ((:math:`N_{query\_points}`) :class:`np.ndarray`):
            A segment array indicating the first bond index for each query
            point.
        neighbor_counts ((:math:`N_{query\_points}`) :class:`np.ndarray`):
            A neighbor count array indicating the number of neighbors for each
            query point.

    Example::

       # Assume we have position as Nx3 array
       aq = freud.locality.AABBQuery(box, positions)
       nlist = aq.query(positions, {'r_max': 3}).toNeighborList()

       # Get all vectors from central particles to their neighbors
       rijs = (positions[nlist.point_indices] -
              positions[nlist.query_point_indices])
       rijs = box.wrap(rijs)

    The NeighborList can be indexed to access bond particle indices. Example::

       for i, j in nlist[:]:
           print(i, j)
    """

    @classmethod
    def from_arrays(cls, num_query_points, num_points, query_point_indices,
                    point_indices, distances, weights=None):
        R"""Create a NeighborList from a set of bond information arrays.

        Args:
            num_query_points (int):
                Number of query points (corresponding to
                :code:`query_point_indices`).
            num_points (int):
                Number of points (corresponding to :code:`point_indices`).
            query_point_indices (:class:`np.ndarray`):
                Array of integers corresponding to indices in the set of
                query points.
            point_indices (:class:`np.ndarray`):
                Array of integers corresponding to indices in the set of
                points.
            distances (:class:`np.ndarray`):
                Array of distances between corresponding query points and
                points.
            weights (:class:`np.ndarray`, optional):
                Array of per-bond weights (if :code:`None` is given, use a
                value of 1 for each weight) (Default value = :code:`None`).
        """
        query_point_indices = freud.common.convert_array(
            query_point_indices, shape=(None,), dtype=np.uint32)
        point_indices = freud.common.convert_array(
            point_indices, shape=query_point_indices.shape, dtype=np.uint32)

        distances = freud.common.convert_array(
            distances, shape=query_point_indices.shape)

        if weights is None:
            weights = np.ones(query_point_indices.shape, dtype=np.float32)
        weights = freud.common.convert_array(
            weights, shape=query_point_indices.shape)

        cdef const unsigned int[::1] l_query_point_indices = \
            query_point_indices
        cdef const unsigned int[::1] l_point_indices = point_indices
        cdef const float[::1] l_distances = distances
        cdef const float[::1] l_weights = weights
        cdef unsigned int l_num_bonds = l_query_point_indices.shape[0]
        cdef unsigned int l_num_query_points = num_query_points
        cdef unsigned int l_num_points = num_points

        cdef NeighborList result
        result = cls()
        result.thisptr = new freud._locality.NeighborList(
            l_num_bonds, &l_query_point_indices[0], l_num_query_points,
            &l_point_indices[0], l_num_points, &l_distances[0], &l_weights[0])

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

    cdef freud._locality.NeighborList * get_ptr(self):
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

    def __getitem__(self, key):
        R"""Access the bond array by index or slice."""
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getNeighbors(),
            freud.util.arr_type_t.UNSIGNED_INT)[key]

    @property
    def query_point_indices(self):
        return self[:, 0]

    @property
    def point_indices(self):
        return self[:, 1]

    @property
    def weights(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getWeights(),
            freud.util.arr_type_t.FLOAT)

    @property
    def distances(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getDistances(),
            freud.util.arr_type_t.FLOAT)

    @property
    def segments(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getSegments(),
            freud.util.arr_type_t.UNSIGNED_INT)

    @property
    def neighbor_counts(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getCounts(),
            freud.util.arr_type_t.UNSIGNED_INT)

    def __len__(self):
        R"""Returns the number of bonds stored in this object."""
        return self.thisptr.getNumBonds()

    def find_first_index(self, unsigned int i):
        R"""Returns the lowest bond index corresponding to a query particle
        with an index :math:`\geq i`.

        Args:
            i (unsigned int): The particle index.
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
            nlist.filter(types[nlist.query_point_indices] != types[nlist.point_indices])
        """  # noqa E501
        filt = np.ascontiguousarray(filt, dtype=np.bool)
        cdef np.ndarray[np.uint8_t, ndim=1, cast=True] filt_c = filt
        cdef cbool * filt_ptr = <cbool*> filt_c.data
        self.thisptr.filter(filt_ptr)
        return self

    def filter_r(self, float r_max, float r_min=0):
        R"""Removes bonds that are outside of a given radius range.

        Args:
            r_max (float):
                Maximum bond distance in the resulting neighbor list.
            r_min (float, optional):
                Minimum bond distance in the resulting neighbor list
                (Default value = :code:`0`).
        """
        self.thisptr.filter_r(r_max, r_min)
        return self


cdef class NlistptrWrapper:
    R"""Wrapper class to hold :code:`freud._locality.NeighborList *`.

    This class is to handle the logic of changing :code:`None` to :code:`NULL`
    in Cython.

    Args:
        nlist (:class:`freud.locality.NeighborList`):
            Neighbor list or :code:`None`.
    """

    def __cinit__(self, nlist):
        cdef NeighborList _nlist
        if nlist is not None:
            _nlist = nlist
            self.nlistptr = _nlist.get_ptr()
        else:
            self.nlistptr = NULL

    cdef freud._locality.NeighborList * get_ptr(self):
        return self.nlistptr


def make_default_nq(box, points):
    R"""Helper function to return a NeighborQuery object.

    Args:
        box (:class:`freud.box.Box`):
            Simulation box.
        points (:class:`freud.locality.AABBQuery`,
            :class:`freud.locality.LinkCell`, or :class:`numpy.ndarray`):
            NeighborQuery object or NumPy array used to build :class:`RawPoints`.

    Returns:
        :class:`freud.locality.NeighborQuery`
            The same :class:`NeighborQuery` object if one is given or :class:`RawPoints`
            built from :code:`box` and :code:`points`.
    """  # noqa: E501
    if isinstance(points, NeighborQuery):
        if points.box != box:
            raise ValueError("The box provided and the box of the"
                             "NeighborQuery object are different")
        return points

    points = freud.common.convert_array(
        points, shape=(None, 3))
    cdef RawPoints rp = RawPoints(box, points)
    return rp


def make_default_nlist(box, points, query_points, query_args, nlist=None):
    R"""Helper function to return a neighbor list object if is given, or to
    construct one using AABBQuery if it is not.

    Args:
        box (:class:`freud.box.Box`):
            Simulation box.
        points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
            Points for the neighborlist.
        query_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
            Query points to construct the neighborlist.
        query_args (dict):
            Query arguments to use. Note that, if it is not one of the provided
            query arguments, :code:`exclude_ii` will be set to :code:`False` if
            query_points is :code:`None` and :code:`True` otherwise.
        nlist (:class:`freud.locality.NeighborList`, optional):
            NeighborList to use to find bonds (Default value = :code:`None`).

    Returns:
        tuple (:class:`freud.locality.NeighborList`, :class:`freud.locality.AABBQuery`):
            The NeighborList and the owning AABBQuery object.
    """  # noqa: E501
    if nlist is not None:
        return nlist

    cdef AABBQuery aq = AABBQuery(box, points)
    query_args.setdefault('exclude_ii', query_points is None)
    cdef _QueryArgs qa = _QueryArgs.from_dict(query_args)
    qp = query_points if query_points is not None else points
    cdef NeighborList aq_nlist = aq.query(
        qp, query_args).toNeighborList()

    return aq_nlist


cdef class RawPoints(NeighborQuery):
    R"""Dummy class that only contains minimal information
    to make C++ side work well.

    .. moduleauthor:: Jin Soo Ihm <jinihm@umich.edu>

    Attributes:
        box (:class:`freud.locality.Box`):
            The simulation box.
        points (:class:`np.ndarray`):
            The points associated with this class.
    """  # noqa: E501

    def __cinit__(self, box, points):
        cdef const float[:, ::1] l_points
        if type(self) is RawPoints:
            # Assume valid set of arguments is passed
            self._box = freud.common.convert_box(box)
            self.points = freud.common.convert_array(
                points, shape=(None, 3))
            l_points = self.points
            self.thisptr = self.nqptr = new freud._locality.RawPoints(
                dereference(self._box.thisptr),
                <vec3[float]*> &l_points[0, 0],
                self.points.shape[0])

    def __dealloc__(self):
        if type(self) is RawPoints:
            del self.thisptr


cdef class AABBQuery(NeighborQuery):
    R"""Use an AABB tree to find neighbors.

    .. moduleauthor:: Bradley Dice <bdice@bradleydice.com>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Attributes:
        box (:class:`freud.locality.Box`):
            The simulation box.
        points (:class:`np.ndarray`):
            The points associated with this class.
    """  # noqa: E501

    def __cinit__(self, box, points):
        cdef const float[:, ::1] l_points
        if type(self) is AABBQuery:
            # Assume valid set of arguments is passed
            self._box = freud.common.convert_box(box)
            self.points = freud.common.convert_array(
                points, shape=(None, 3)).copy()
            l_points = self.points
            self.thisptr = self.nqptr = new freud._locality.AABBQuery(
                dereference(self._box.thisptr),
                <vec3[float]*> &l_points[0, 0],
                self.points.shape[0])

    def __dealloc__(self):
        if type(self) is AABBQuery:
            del self.thisptr


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
        points (:class:`np.ndarray`, optional):
            The points associated with this class, if used as a NeighborQuery
            object, i.e. built on one set of points that can then be queried
            against.  (Default value = :code:`None`).

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

    def __cinit__(self, box, cell_width, points):
        self._box = freud.common.convert_box(box)
        cdef const float[:, ::1] l_points
        self.points = freud.common.convert_array(
            points, shape=(None, 3)).copy()
        l_points = self.points
        self.thisptr = self.nqptr = new freud._locality.LinkCell(
            dereference(self._box.thisptr), float(cell_width),
            <vec3[float]*> &l_points[0, 0],
            self.points.shape[0])

    def __dealloc__(self):
        del self.thisptr

    @property
    def box(self):
        return freud.box.BoxFromCPP(self.thisptr.getBox())

    @property
    def num_cells(self):
        return self.thisptr.getNumCells()

    def getCell(self, point):
        R"""Returns the index of the cell containing the given point.

        Args:
            point(:math:`\left(3\right)` :class:`numpy.ndarray`):
                Point coordinates :math:`\left(x,y,z\right)`.

        Returns:
            unsigned int: Cell index.
        """
        point = freud.common.convert_array(point, shape=(None, ))

        cdef const float[::1] cPoint = point

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


cdef class Voronoi(Compute):
    R"""Compute the Voronoi tessellation of a 2D or 3D system using voro++.

    .. moduleauthor:: Bradley Dice <bdice@bradleydice.com>
    .. moduleauthor:: Yezhi Jin <jinyezhi@umich.com>

    Attributes:
        nlist (:class:`~.locality.NeighborList`):
            Returns a weighted neighbor list. In 2D systems, the bond weight is
            the "ridge length" of the Voronoi boundary line between the
            neighboring particles. In 3D systems, the bond weight is the "ridge
            area" of the Voronoi boundary polygon between the neighboring
            particles.
        polytopes (list[:class:`numpy.ndarray`]):
            List of arrays, each containing Voronoi cell polytope vertices.
        volumes ((:math:`\left(N_{cells} \right)`) :class:`numpy.ndarray`):
            Returns an array of Voronoi cell volumes (areas in 2D).
    """

    def __cinit__(self):
        self.thisptr = new freud._locality.VoroPlusPlus()
        self._nlist = NeighborList()

    def __dealloc__(self):
        del self.thisptr

    @Compute._compute()
    def compute(self, box, points):
        R"""Compute Voronoi diagram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{points}`, 3) :class:`numpy.ndarray`):
                Points used to calculate Voronoi diagram.
        """
        self._box = freud.common.convert_box(box)

        # voro++ uses double precision
        points = freud.common.convert_array(points, shape=(None, 3),
                                            dtype=np.float64)
        cdef const double[:, ::1] l_points = points
        cdef unsigned int N = len(points)

        self.thisptr.compute(
            dereference(self._box.thisptr),
            <vec3[double]*> &l_points[0, 0],
            N)

        cdef freud._locality.NeighborList * nlist
        nlist = self.thisptr.getNeighborList()
        self._nlist.refer_to(nlist)
        self._nlist.base = self

        return self

    @Compute._computed_property()
    def polytopes(self):
        R"""Returns a list of polytope vertices corresponding to Voronoi cells.

        If the buffer width is too small, then some polytopes may not be
        closed (they may have a boundary at infinity), and these polytopes'
        vertices are excluded from the list.

        The length of the list returned by this method should be the same
        as the array of positions used in the
        :meth:`freud.locality.Voronoi.compute()` method, if all the polytopes
        are closed. Otherwise try using a larger buffer width.

        Returns:
            list:
                List of :class:`numpy.ndarray` containing Voronoi polytope
                vertices.
        """
        polytopes = []
        cdef vector[vector[vec3[double]]] raw_polytopes = \
            self.thisptr.getPolytopes()
        cdef size_t i
        cdef size_t j
        cdef size_t num_verts
        cdef vector[vec3[double]] raw_vertices
        cdef vec3[double] vertex
        cdef double[:, ::1] polytope_vertices
        for i in range(raw_polytopes.size()):
            raw_vertices = raw_polytopes[i]
            num_verts = raw_vertices.size()
            polytope_vertices = np.empty((num_verts, 3), dtype=np.float64)
            for j in range(num_verts):
                polytope_vertices[j, 0] = raw_vertices[j].x
                polytope_vertices[j, 1] = raw_vertices[j].y
                polytope_vertices[j, 2] = raw_vertices[j].z
            polytopes.append(np.asarray(polytope_vertices))
        return polytopes

    @Compute._computed_property()
    def nlist(self):
        R"""Returns a neighbor list object.

        In the neighbor list, each neighbor pair has a weight value.

        In 2D systems, the bond weight is the "ridge length" of the Voronoi
        boundary line between the neighboring particles.

        In 3D systems, the bond weight is the "ridge area" of the Voronoi
        boundary polygon between the neighboring particles.

        Returns:
            :class:`~.locality.NeighborList`: Neighbor list.
        """
        return self._nlist

    @Compute._computed_property()
    def volumes(self):
        R"""Returns an array of volumes (areas in 2D) corresponding to Voronoi
        cells.

        Returns:
            (:math:`\left(N_{cells} \right)`) :class:`numpy.ndarray`:
                Array of voronoi polytope volumes (areas in 2D).
        """
        return np.asarray(self.thisptr.getVolumes())

    def __repr__(self):
        return "freud.locality.{cls}()".format(
            cls=type(self).__name__)

    def __str__(self):
        return repr(self)

    @Compute._computed_method()
    def plot(self, ax=None):
        """Plot Voronoi diagram.

        Args:
            ax (:class:`matplotlib.axes.Axes`): Axis to plot on. If
                :code:`None`, make a new figure and axis.
                (Default value = :code:`None`)

        Returns:
            (:class:`matplotlib.axes.Axes`): Axis with the plot.
        """
        import plot
        if not self._box.is2D():
            return None
        else:
            return plot.draw_voronoi(self._box, self.polytopes, ax=ax)

    def _repr_png_(self):
        import plot
        try:
            return plot.ax_to_bytes(self.plot())
        except AttributeError:
            return None
