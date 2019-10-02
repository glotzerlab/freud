# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The :mod:`freud.locality` module contains data structures to efficiently
locate points based on their proximity to other points.
"""
import sys
import numpy as np
import itertools
import warnings
import logging
import copy
import freud.util

from libcpp cimport bool as cbool
from freud.util cimport vec3
from cython.operator cimport dereference
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from freud._locality cimport ITERATOR_TERMINATOR
from freud.util cimport Compute

cimport freud._locality
cimport freud.box
cimport numpy as np

logger = logging.getLogger(__name__)

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

    def __repr__(self):
        return ("freud.locality.{cls}(mode={mode}, r_max={r_max}, "
                "num_neighbors={num_neighbors}, exclude_ii={exclude_ii}, "
                "scale={scale})").format(
                    cls=type(self).__name__,
                    mode=self.mode, r_max=self.r_max,
                    num_neighbors=self.num_neighbors,
                    exclude_ii=self.exclude_ii,
                    scale=self.scale)

    def __str__(self):
        return repr(self)


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
            yield (npoint.query_point_idx, npoint.point_idx, npoint.distance)
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
        cdef NeighborList nl = _nlist_from_cnlist(cnlist)
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

    @classmethod
    def from_system(cls, system):
        R"""Create a :class:`~.NeighborQuery` from any system-like object.

        The standard concept of a system in **freud** is any object that
        provides a way to access a box-like object (anything that can be
        coerced to a box by :meth:`freud.box.Box.box`) and an array-like
        (according to `NumPy's definition
        <https://docs.scipy.org/doc/numpy/user/basics.creation.html#converting-python-array-like-objects-to-numpy-arrays>`_)
        object that turns into a :math:`N\times 3` array.

        Supported types for :code:`system` include:
        - :class:`~.locality.AABBQuery`
        - :class:`~.locality.LinkCell`
        - :class:`~.locality.RawPoints`
        - A sequence of :code:`(box, points)` where :code:`box` is a
          :class:`~.box.Box` and :code:`points` is a :class:`numpy.ndarray`.

        Args:
            system (system-like object):
                Any object that can be converted to a :class:`~.NeighborQuery`.

        Returns:
            :class:`freud.locality.NeighborQuery`
                The same :class:`NeighborQuery` object if one is given or
                :class:`RawPoints` built from an inferred :code:`box` and
                :code:`points`.
        """
        if isinstance(system, NeighborQuery):
            return system
        elif cls == NeighborQuery:
            # If called from this abstract parent class, always make
            # :class:`~.RawPoints`.
            return RawPoints(*system)
        else:
            # Otherwise, use the current class.
            return cls(*system)
        raise ValueError(
            "This system could not be coerced into a NeighborQuery object.")

    @property
    def box(self):
        return freud.box.BoxFromCPP(self.nqptr.getBox())

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
        query_points = freud.util._convert_array(
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
        query_point_indices = freud.util._convert_array(
            query_point_indices, shape=(None,), dtype=np.uint32)
        point_indices = freud.util._convert_array(
            point_indices, shape=query_point_indices.shape, dtype=np.uint32)

        distances = freud.util._convert_array(
            distances, shape=query_point_indices.shape)

        if weights is None:
            weights = np.ones(query_point_indices.shape, dtype=np.float32)
        weights = freud.util._convert_array(
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

    def __cinit__(self, _null=False):
        # Setting _null to True will create a NeighborList with no underlying
        # C++ object. This is useful for passing NULL pointers to C++ to
        # indicate the lack of a NeighborList
        self._managed = not _null
        # Cython won't assign NULL without cast
        self.thisptr = <freud._locality.NeighborList *> NULL if _null \
            else new freud._locality.NeighborList()

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
        cdef cbool * filt_ptr = <cbool*> &filt_c[0]
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


cdef NeighborList _nlist_from_cnlist(freud._locality.NeighborList *c_nlist):
    """Create a Python NeighborList object that points to an existing C++
    NeighborList object.

    This functions generally serves two purposes. Any special locality
    NeighborList generators, like :class:`~.Voronoi`, should use this as a way
    to point to the C++ NeighborList they generate internally. Additionally,
    any compute method that requires a :class:`~.NeighborList` (i.e. cannot do
    with just a :class:`~.NeighborQuery`) should also expose the internally
    computed :class:`~.NeighborList` using this method.
    """
    cdef NeighborList result
    result = NeighborList()
    del result.thisptr
    result._managed = False
    result.thisptr = c_nlist
    return result


def _make_default_nlist(system, query_points, query_args, nlist=None):
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
        :class:`freud.locality.NeighborList`:
            The neighbor list.
    """  # noqa: E501
    if nlist is not None:
        return nlist

    cdef NeighborQuery nq = NeighborQuery.from_system(system)
    query_args.setdefault('exclude_ii', query_points is None)
    qp = query_points if query_points is not None else nq.points
    cdef NeighborList nq_nlist = nq.query(qp, query_args).toNeighborList()

    return nq_nlist


cdef class RawPoints(NeighborQuery):
    R"""Dummy class that only contains minimal information
    to make C++ side work well.

    Attributes:
        box (:class:`freud.locality.Box`):
            The simulation box.
        points (:class:`np.ndarray`):
            The points associated with this class.
    """  # noqa: E501

    def __cinit__(self, box, points):
        cdef const float[:, ::1] l_points
        cdef freud.box.Box b
        if type(self) is RawPoints:
            # Assume valid set of arguments is passed
            b = freud.util._convert_box(box)
            self.points = freud.util._convert_array(
                points, shape=(None, 3))
            l_points = self.points
            self.thisptr = self.nqptr = new freud._locality.RawPoints(
                dereference(b.thisptr),
                <vec3[float]*> &l_points[0, 0],
                self.points.shape[0])

    def __dealloc__(self):
        if type(self) is RawPoints:
            del self.thisptr


cdef class AABBQuery(NeighborQuery):
    R"""Use an AABB tree to find neighbors.

    Attributes:
        box (:class:`freud.locality.Box`):
            The simulation box.
        points (:class:`np.ndarray`):
            The points associated with this class.
    """  # noqa: E501

    def __cinit__(self, box, points):
        cdef const float[:, ::1] l_points
        cdef freud.box.Box b
        if type(self) is AABBQuery:
            # Assume valid set of arguments is passed
            b = freud.util._convert_box(box)
            self.points = freud.util._convert_array(
                points, shape=(None, 3)).copy()
            l_points = self.points
            self.thisptr = self.nqptr = new freud._locality.AABBQuery(
                dereference(b.thisptr),
                <vec3[float]*> &l_points[0, 0],
                self.points.shape[0])

    def __dealloc__(self):
        if type(self) is AABBQuery:
            del self.thisptr


cdef class LinkCell(NeighborQuery):
    R"""Supports efficiently finding all points in a set within a certain
    distance from a given point.

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
    """

    def __cinit__(self, box, points, cell_width=0):
        cdef freud.box.Box b = freud.util._convert_box(box)
        cdef const float[:, ::1] l_points
        self.points = freud.util._convert_array(
            points, shape=(None, 3)).copy()
        l_points = self.points
        self.thisptr = self.nqptr = new freud._locality.LinkCell(
            dereference(b.thisptr),
            <vec3[float]*> &l_points[0, 0],
            self.points.shape[0], cell_width)

    def __dealloc__(self):
        del self.thisptr

    @property
    def cell_width(self):
        return self.thisptr.getCellWidth()


cdef class Voronoi(Compute):
    R"""Computes Voronoi diagrams using voro++.

    Voronoi diagrams (`Wikipedia
    <https://en.wikipedia.org/wiki/Voronoi_diagram>`_) are composed of convex
    polytopes (polyhedra in 3D, polygons in 2D) called cells, corresponding to
    each input point. The cells bound a region of Euclidean space for which all
    contained points are closer to a corresponding input point than any other
    input point. A ridge is defined as a boundary between cells, which contains
    points equally close to two or more input points.

    The voro++ library [Rycroft2009]_ is used for fast computations of the
    Voronoi diagram.

    .. [Rycroft2009] Rycroft, Chris (2009). Voro++: a three-dimensional Voronoi
       cell library in C++. Technical Report. https://doi.org/10.2172/946741

    Attributes:
        nlist (:class:`~.locality.NeighborList`):
            Returns a neighbor list weighted by ridge area (length in 2D).
        polytopes (list[:class:`numpy.ndarray`]):
            A list of :class:`numpy.ndarray` defining Voronoi polytope vertices
            for each cell.
        volumes (:math:`\left(N_{points} \right)` :class:`numpy.ndarray`):
            Returns an array of Voronoi cell volumes (areas in 2D).
    """

    def __cinit__(self):
        self.thisptr = new freud._locality.Voronoi()
        self._nlist = NeighborList()

    def __dealloc__(self):
        del self.thisptr

    def compute(self, system):
        R"""Compute Voronoi diagram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{points}`, 3) :class:`numpy.ndarray`):
                Points used to calculate Voronoi diagram.
        """
        cdef NeighborQuery nq = NeighborQuery.from_system(system)
        self.thisptr.compute(nq.get_ptr())
        return self

    @Compute._computed_property
    def polytopes(self):
        R"""Polytope vertices of each Voronoi cell.

        Returns:
            list:
                List of :class:`numpy.ndarray` defining Voronoi polytope
                vertices for each cell.
        """
        polytopes = []
        cdef vector[vector[vec3[double]]] raw_polytopes = \
            self.thisptr.getPolytopes()
        cdef size_t i
        cdef size_t j
        cdef size_t num_verts
        cdef vector[vec3[double]] raw_vertices
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

    @Compute._computed_property
    def volumes(self):
        R"""Returns an array of volumes (areas in 2D) of the Voronoi cells.

        Returns:
            :math:`\left(N_{points} \right)` :class:`numpy.ndarray`:
                Array of Voronoi polytope volumes (areas in 2D).
        """
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getVolumes(),
            freud.util.arr_type_t.DOUBLE)

    @Compute._computed_property
    def nlist(self):
        R"""Returns the computed :class:`~.locality.NeighborList`.

        The :class:`~.locality.NeighborList` computed by this class is
        weighted. In 2D systems, the bond weight is the length of the ridge
        (boundary line) between the neighboring points' Voronoi cells. In 3D
        systems, the bond weight is the area of the ridge (boundary polygon)
        between the neighboring points' Voronoi cells. The weights are not
        normalized, and the weights for each query point sum to the surface
        area (perimeter in 2D) of the polytope.

        It is possible for pairs of points to appear multiple times in the
        neighbor list. For example, in a small unit cell, points may neighbor
        one another on multiple sides because of periodic boundary conditions.

        Returns:
            :class:`~.locality.NeighborList`: Neighbor list.
        """
        self._nlist = _nlist_from_cnlist(self.thisptr.getNeighborList().get())
        return self._nlist

    def __repr__(self):
        return "freud.locality.{cls}()".format(
            cls=type(self).__name__)

    def __str__(self):
        return repr(self)

    def plot(self, ax=None):
        """Plot Voronoi diagram.

        Args:
            ax (:class:`matplotlib.axes.Axes`): Axis to plot on. If
                :code:`None`, make a new figure and axis.
                (Default value = :code:`None`)

        Returns:
            :class:`matplotlib.axes.Axes`: Axis with the plot.
        """
        import freud.plot
        if not self._box.is2D:
            return None
        else:
            return freud.plot.voronoi_plot(self._box, self.polytopes, ax=ax)

    def _repr_png_(self):
        import freud.plot
        try:
            return freud.plot.ax_to_bytes(self.plot())
        except AttributeError:
            return None


cdef class PairCompute(Compute):
    R"""Parent class for all compute classes in freud that depend on finding
    nearest neighbors.

    The purpose of this class is to consolidate some of the logic for parsing
    the numerous possible inputs to the compute calls of such classes. In
    particular, this class contains a helper function that calls the necessary
    functions to create NeighborQuery and NeighborList classes as needed, as
    well as dealing with boxes and query arguments.
    """

    def _preprocess_arguments(self, system, query_points=None,
                              neighbors=None):
        """Process standard compute arguments into freud's internal types by
        calling all the required internal functions.

        This function handles the preprocessing of boxes and points into
        :class:`freud.locality.NeighborQuery` objects, the determination of how
        to handle the NeighborList object, the creation of default query
        arguments as needed, deciding what `query_points` are, and setting the
        appropriate `exclude_ii` flag.

        Args:
            system (:class:`freud.locality.NeighborQuery` or tuple):
                If a tuple, must be of the form (box_like, array_like), i.e. it
                must be an object that can be converted into a
                :class:`freud.locality.NeighborQuery`.
            query_points ((:math:`N_{query\_points}`, 3) :class:`numpy.ndarray`, optional):
                Points used to calculate the RDF. Uses :code:`points` if
                not provided or :code:`None`.
            neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                NeighborList or dictionary of query arguments to use to find
                bonds (Default value = :code:`None`).
        """  # noqa E501
        cdef NeighborQuery nq = NeighborQuery.from_system(system)

        # Resolve the two possible ways of passing neighbors (query arguments
        # or neighbor lists) based on the type of the neighbors argument.
        cdef NeighborList nlist
        cdef _QueryArgs qargs

        nlist, qargs = self._resolve_neighbors(neighbors, query_points)

        if query_points is None:
            query_points = nq.points
        else:
            query_points = freud.util._convert_array(
                query_points, shape=(None, 3))
        cdef const float[:, ::1] l_query_points = query_points
        cdef unsigned int num_query_points = l_query_points.shape[0]
        return (nq, nlist, qargs, l_query_points, num_query_points)

    def _resolve_neighbors(self, neighbors, query_points=None):
        if type(neighbors) == NeighborList:
            nlist = neighbors
            qargs = _QueryArgs()
        elif neighbors is None or type(neighbors) == dict:
            # The default_query_args property must raise a NotImplementedError
            # if no query arguments were passed in and the class has no
            # reasonable choice of defaults.
            try:
                query_args = self.default_query_args if neighbors is None \
                    else neighbors.copy()
                query_args.setdefault('exclude_ii', query_points is None)
                qargs = _QueryArgs.from_dict(query_args)
                nlist = NeighborList(True)
            except NotImplementedError:
                raise
        return nlist, qargs

    @property
    def default_query_args(self):
        raise NotImplementedError(
            "The {} class does not provide default query arguments. You must "
            "either provide query arguments or a neighbor list to this "
            "compute method.".format(type(self).__name__))


cdef class SpatialHistogram(PairCompute):
    R"""Parent class for all compute classes in freud that perform a spatial
    binning of particle bonds by distance.
    """

    def __cinit__(self):
        # Abstract class
        pass

    @property
    def default_query_args(self):
        return dict(mode="ball", r_max=self.r_max)

    @Compute._computed_property
    def box(self):
        return freud.box.BoxFromCPP(self.histptr.getBox())

    @Compute._computed_property
    def bin_counts(self):
        return freud.util.make_managed_numpy_array(
            &self.histptr.getBinCounts(),
            freud.util.arr_type_t.UNSIGNED_INT)

    @property
    def bin_centers(self):
        # Must create a local reference or Cython tries to access an rvalue by
        # reference in the list comprehension.
        vec = self.histptr.getBinCenters()
        return [np.array(b, copy=True) for b in vec]

    @property
    def bin_edges(self):
        # Must create a local reference or Cython tries to access an rvalue by
        # reference in the list comprehension.
        vec = self.histptr.getBinEdges()
        return [np.array(b, copy=True) for b in vec]

    @property
    def bounds(self):
        # Must create a local reference or Cython tries to access an rvalue by
        # reference in the list comprehension.
        vec = self.histptr.getBounds()
        return [tuple(b) for b in vec]

    @property
    def nbins(self):
        return list(self.histptr.getAxisSizes())

    def _reset(self):
        # Resets the values of RDF in memory.
        self.histptr.reset()


cdef class SpatialHistogram1D(SpatialHistogram):
    R"""Subclasses SpatialHistogram to provide a simplified API for
    properties of 1-dimensional histograms.
    """

    def __cinit__(self):
        # Abstract class
        pass

    @property
    def bin_centers(self):
        # Must create a local reference or Cython tries to access an rvalue by
        # reference in the list comprehension.
        vec = self.histptr.getBinCenters()
        return np.array(vec[0], copy=True)

    @property
    def bin_edges(self):
        # Must create a local reference or Cython tries to access an rvalue by
        # reference in the list comprehension.
        vec = self.histptr.getBinEdges()
        return np.array(vec[0], copy=True)

    @property
    def bounds(self):
        # Must create a local reference or Cython tries to access an rvalue by
        # reference in the list comprehension.
        vec = self.histptr.getBounds()
        return vec[0]

    @property
    def nbins(self):
        return self.histptr.getAxisSizes()[0]
