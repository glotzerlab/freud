# Copyright (c) 2010-2026 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

r"""
The :mod:`freud.locality` module contains data structures to efficiently
locate points based on their proximity to other points.
"""

import inspect
from importlib.util import find_spec

import numpy as np

import freud._locality
import freud.box
import freud.util
from freud._util import (  # noqa F401
    ManagedArray_double,
    ManagedArray_float,
    ManagedArray_unsignedint,
    ManagedArrayVec3_float,
    Vector_double,
    Vector_float,
    Vector_unsignedint,
    VectorVec3_float,
)
from freud.errors import NO_DEFAULT_QUERY_ARGS_MESSAGE
from freud.util import _Compute

_HAS_MPL = find_spec("matplotlib") is not None
if _HAS_MPL:
    import freud.plot
else:
    msg_mpl = "Plotting requires matplotlib."

ITERATOR_TERMINATOR = freud._locality.get_iterator_terminator()


class _QueryArgs:
    r"""Container for query arguments.

    This class is use internally throughout freud to provide a nice interface
    between keyword- or dict-style query arguments and the C++ QueryArgs
    object. All arguments are funneled through this interface, which constructs
    the appropriate C++ QueryArgs object that can then be used in C++ compute
    calls.
    """

    def __init__(
        self,
        mode=None,
        r_min=None,
        r_max=None,
        r_guess=None,
        num_neighbors=None,
        exclude_ii=None,
        scale=None,
        **kwargs,
    ):
        self._cpp_obj = freud._locality.QueryArgs()
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
        if len(kwargs) > 0:
            err_str = ", ".join(f"{k} = {v}" for k, v in kwargs.items())
            raise ValueError(
                "The following invalid query arguments were provided: " + err_str
            )

    def update(self, qargs):
        if qargs is None:
            return
        for arg in qargs:
            if hasattr(self, arg):
                setattr(self, arg, qargs[arg])
            else:
                msg = "An invalid query argument was provided."
                raise ValueError(msg)

    @classmethod
    def from_dict(cls, mapping):
        """Create _QueryArgs from mapping."""
        return cls(**mapping)

    @property
    def mode(self):
        if self._cpp_obj.mode == freud._locality.QueryType.none:
            return None
        if self._cpp_obj.mode == freud._locality.QueryType.ball:
            return "ball"
        if self._cpp_obj.mode == freud._locality.QueryType.nearest:
            return "nearest"
        msg = f"Unknown mode {self._cpp_obj.mode} set!"
        raise ValueError(msg)

    @mode.setter
    def mode(self, value):
        if value == "none" or value is None:
            self._cpp_obj.mode = freud._locality.QueryType.none
        elif value == "ball":
            self._cpp_obj.mode = freud._locality.QueryType.ball
        elif value == "nearest":
            self._cpp_obj.mode = freud._locality.QueryType.nearest
        else:
            msg = "An invalid mode was provided."
            raise ValueError(msg)

    @property
    def r_guess(self):
        return self._cpp_obj.r_guess

    @r_guess.setter
    def r_guess(self, value):
        self._cpp_obj.r_guess = value

    @property
    def r_min(self):
        return self._cpp_obj.r_min

    @r_min.setter
    def r_min(self, value):
        self._cpp_obj.r_min = value

    @property
    def r_max(self):
        return self._cpp_obj.r_max

    @r_max.setter
    def r_max(self, value):
        self._cpp_obj.r_max = value

    @property
    def num_neighbors(self):
        return self._cpp_obj.num_neighbors

    @num_neighbors.setter
    def num_neighbors(self, value):
        self._cpp_obj.num_neighbors = value

    @property
    def exclude_ii(self):
        return self._cpp_obj.exclude_ii

    @exclude_ii.setter
    def exclude_ii(self, value):
        self._cpp_obj.exclude_ii = value

    @property
    def scale(self):
        return self._cpp_obj.scale

    @scale.setter
    def scale(self, value):
        self._cpp_obj.scale = value

    def __repr__(self):
        return (
            f"freud.locality.{type(self).__name__}"
            f"(mode={self.mode}, r_max={self.r_max}, "
            f"num_neighbors={self.num_neighbors}, exclude_ii={self.exclude_ii}, "
            f"scale={self.scale})"
        )

    def __str__(self):
        return repr(self)


class NeighborQueryResult:
    r"""Class encapsulating the output of queries of NeighborQuery objects.

    .. warning::

        This class should not be instantiated directly, it is the
        return value of the :meth:`~NeighborQuery.query` method of
        :class:`~NeighborQuery`. The class provides a convenient
        interface for iterating over query results, and can be
        transparently converted into a list or a
        :class:`~NeighborList` object.

    The :class:`~NeighborQueryResult` makes it easy to work with the results of
    queries and convert them to various natural objects. Additionally, the
    result is a generator, making it easy for users to lazily iterate over the
    object.
    """

    def __init__(self, nq, points, query_args):
        self._nq = nq
        self._points = np.array(points, dtype=np.float32)
        self._query_args = query_args

    def __iter__(self):
        iterator_cpp = self._nq._cpp_obj.query(self._points, self._query_args._cpp_obj)

        npoint = iterator_cpp.next()
        while npoint != ITERATOR_TERMINATOR:
            yield (
                npoint.getQueryPointIdx(),
                npoint.getPointIdx(),
                npoint.getDistance(),
            )
            npoint = iterator_cpp.next()

    def toNeighborList(self, sort_by_distance=False):
        """Convert query result to a freud :class:`~NeighborList`.

        Args:
            sort_by_distance (bool):
                If :code:`True`, sort neighboring bonds by distance.
                If :code:`False`, sort neighboring bonds by point index
                (Default value = :code:`False`).

        Returns:
            :class:`~NeighborList`: A :class:`~NeighborList` containing all
            neighbor pairs found by the query generating this result object.
        """
        iterator_cpp = self._nq._cpp_obj.query(self._points, self._query_args._cpp_obj)

        nlist_cpp = iterator_cpp.toNeighborList(sort_by_distance)
        return _nlist_from_cnlist(nlist_cpp)


class NeighborQuery:
    r"""Class representing a set of points along with the ability to query for
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
    """

    def __init__(self):
        msg = (
            "The NeighborQuery class is abstract, and should not be "
            "directly instantiated"
        )
        raise RuntimeError(msg)

    @classmethod
    def from_system(cls, system, dimensions=None):
        r"""Create a :class:`~.NeighborQuery` from any system-like object.

        The standard concept of a system in **freud** is any object that
        provides a way to access a box-like object (anything that can be
        coerced to a box by :meth:`freud.box.Box.from_box`) and an array-like
        object (according to `NumPy's definition
        <https://docs.scipy.org/doc/numpy/user/basics.creation.html#converting-python-array-like-objects-to-numpy-arrays>`_)
        of particle positions that turns into a :math:`N\times 3` array.

        Supported types for :code:`system` include:

        * :class:`~.locality.AABBQuery`
        * :class:`~.locality.LinkCell`
        * A sequence of :code:`(box, points)` where :code:`box` is a
          :class:`~.box.Box` and :code:`points` is a :class:`numpy.ndarray`.
        * Objects with attributes :code:`box` and :code:`points`.
        * :class:`MDAnalysis.coordinates.timestep.Timestep`
        * :class:`gsd.hoomd.Snapshot`
        * :class:`gsd.hoomd.Frame`
        * :class:`garnett.trajectory.Frame`
        * :class:`ovito.data.DataCollection`
        * :class:`hoomd.Snapshot`

        Args:
            system (system-like object):
                Any object that can be converted to a :class:`~.NeighborQuery`.
            dimensions (int):
                Whether the object is 2 or 3 dimensional. It may be inferred if
                not provided, but in some cases inference is not possible, in
                which case it will default to 3 (Default value = None).

        Returns:
            :class:`freud.locality.NeighborQuery`:
                The same :class:`~.NeighborQuery` object if one is given, or an
                instance of :class:`~.NeighborQuery` built from an inferred
                :code:`box` and :code:`points`.
        """

        def _match_class_path(obj, *matches):
            return any(
                cls.__module__ + "." + cls.__name__ in matches
                for cls in inspect.getmro(type(obj))
            )

        if isinstance(system, cls):
            return system

        # MDAnalysis compatibility
        # base namespace for mdanalysis<2.3.0
        # timestep namespace for mdanalysis>=2.3.0
        if _match_class_path(
            system,
            "MDAnalysis.coordinates.base.Timestep",
            "MDAnalysis.coordinates.timestep.Timestep",
        ):
            system = (system.triclinic_dimensions, system.positions)

        # GSD and HOOMD-blue 3 snapshot compatibility
        elif _match_class_path(
            system, "gsd.hoomd.Frame", "gsd.hoomd.Snapshot", "hoomd.snapshot.Snapshot"
        ):
            # Explicitly construct the box to silence warnings from box
            # constructor, HOOMD simulations often have Lz=1 for 2D boxes.
            box = np.array(system.configuration.box)
            if system.configuration.dimensions == 2:
                box[[2, 4, 5]] = 0
            system = (box, system.particles.position)

        # garnett compatibility (garnett >=0.5)
        elif _match_class_path(system, "garnett.trajectory.Frame"):
            try:
                # garnett >= 0.7
                position = system.position
            except AttributeError:
                # garnett < 0.7
                position = system.positions
            system = (system.box, position)

        # OVITO compatibility
        elif _match_class_path(
            system,
            "ovito.data.DataCollection",
            "ovito.plugins.PyScript.DataCollection",
            "PyScript.DataCollection",
        ):
            box = freud.Box.from_box(
                system.cell.matrix[:, :3], dimensions=2 if system.cell.is2D else 3
            )
            system = (box, system.particles.positions)

        # HOOMD-blue 2 snapshot compatibility
        elif (
            hasattr(system, "box")
            and hasattr(system, "particles")
            and hasattr(system.particles, "position")
        ):
            # Explicitly construct the box to silence warnings from box
            # constructor because HOOMD sets Lz=1 rather than 0 for 2D boxes.
            if system.box.dimensions == 2:
                box = freud.Box(system.box.Lx, system.box.Ly, xy=system.box.xy)
            else:
                box = system.box
            system = (box, system.particles.position)

        # Duck type systems with attributes into a (box, points) tuple
        elif hasattr(system, "box") and hasattr(system, "points"):
            system = (system.box, system.points)

        if cls == NeighborQuery:
            # If called from this abstract parent class, always make
            # :class:`~._RawPoints`.
            return _RawPoints(*system)
        # Otherwise, use the current class.
        return cls(*system)

    @property
    def box(self):
        """:class:`freud.box.Box`: The box object used by this data
        structure."""
        return freud.box.BoxFromCPP(self._cpp_obj.getBox())

    @property
    def points(self):
        """:class:`np.ndarray`: The array of points in this data structure."""
        return np.asarray(self._points)

    def query(self, query_points, query_args):
        r"""Query for nearest neighbors of the provided point.

        Args:
            query_points ((:math:`N`, 3) :class:`numpy.ndarray`):
                Points to query for.
            query_args (dict):
                Query arguments determining how to find neighbors. For
                information on valid query argument, see the `Query API
                <https://freud.readthedocs.io/en/stable/topics/querying.html>`_.

        Returns:
            :class:`~.NeighborQueryResult`: Results object containing the
            output of this query.
        """
        query_points = freud.util._convert_array(
            np.atleast_2d(query_points), shape=(None, 3)
        )
        args = _QueryArgs.from_dict(query_args)
        return NeighborQueryResult(self, query_points, args)

    def plot(self, ax=None, title=None, *args, **kwargs):
        """Plot system box and points.

        Args:
            ax (:class:`matplotlib.axes.Axes`):
                Axis to plot on. If :code:`None`, make a new figure and axis.
                The axis projection (2D or 3D) must match the dimensionality of
                the system (Default value = :code:`None`).
            title (str):
                Title of the plot (Default value = :code:`None`).
            *args:
                Passed on to :meth:`mpl_toolkits.mplot3d.Axes3D.plot` or
                :meth:`matplotlib.axes.Axes.plot`.
            **kwargs:
                Passed on to :meth:`mpl_toolkits.mplot3d.Axes3D.plot` or
                :meth:`matplotlib.axes.Axes.plot`.

        Returns:
            tuple (:class:`matplotlib.axes.Axes`, \
            :class:`matplotlib.collections.PathCollection`):
                Axis and point data for the plot.
        """
        if not _HAS_MPL:
            raise ImportError(msg_mpl)
        return freud.plot.system_plot(self, ax=ax, title=title, *args, **kwargs)  # noqa: B026 - it works


class NeighborList:
    r"""Class representing bonds between two sets of points.

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

    Also available as ``freud.NeighborList``.

    Example::

       # Assume we have position as Nx3 array
       aq = freud.locality.AABBQuery(box, positions)
       nlist = aq.query(positions, {'r_max': 3}).toNeighborList()

       # Get all vectors from central particles to their neighbors
       rijs = nlist.vectors
       rijs = box.wrap(rijs)

    The NeighborList can be indexed to access bond particle indices. Example::

       for i, j in nlist[:]:
           print(i, j)
    """

    @classmethod
    def from_arrays(
        cls,
        num_query_points,
        num_points,
        query_point_indices,
        point_indices,
        vectors,
        weights=None,
    ):
        r"""Create a NeighborList from a set of bond information arrays.

        Example::

            import freud
            import numpy as np
            box = freud.box.Box(2, 3, 4, 0, 0, 0)
            query_points = np.array([[0, 0, 0], [0, 0, 1]])
            points = np.array([[0, 0, -1], [0.5, -1, 0]])
            num_query_points = len(query_points)
            num_points = len(points)
            query_point_indices = np.array([0, 0, 1])
            point_indices = np.array([0, 1, 1])
            vectors = box.wrap(points[point_indices] - query_points[query_point_indices])
            nlist = freud.locality.NeighborList.from_arrays(
                num_query_points, num_points, query_point_indices,
                point_indices, vectors)


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
            vectors (:math:`\left(N_{bonds}, 3\right)` :class:`numpy.ndarray`):
                Array of bond vectors from query points to corresponding
                points.
            weights (:math:`\left(N_{bonds} \right)` :class:`np.ndarray`, optional):
                Array of per-bond weights (if :code:`None` is given, use a
                value of 1 for each weight) (Default value = :code:`None`).
        """  # noqa 501
        query_point_indices = freud.util._convert_array(
            query_point_indices, shape=(None,), dtype=np.uint32
        ).copy()
        point_indices = freud.util._convert_array(
            point_indices, shape=query_point_indices.shape, dtype=np.uint32
        ).copy()

        vectors = freud.util._convert_array(
            vectors, shape=(len(query_point_indices), 3), dtype=np.float32
        ).copy()

        if weights is None:
            weights = np.ones(query_point_indices.shape, dtype=np.float32)
        weights = freud.util._convert_array(
            weights, shape=query_point_indices.shape
        ).copy()

        result = cls()
        result._cpp_obj = freud._locality.NeighborList(
            query_point_indices,
            num_query_points,
            point_indices,
            num_points,
            vectors,
            weights,
        )

        return result

    @classmethod
    def all_pairs(cls, system, query_points=None, exclude_ii=True):
        R"""Create a NeighborList where all pairs of points are neighbors.

        More explicitly, this method returns a NeighborList in which all pairs of
        points :math:`i`, :math:`j` are neighbors. Pairs such that :math:`i = j`
        can also be excluded using the ``exclude_ii`` option. The weight of all
        neighbors pairs in the returned list will be 1.

        Args:
            system:
                Any object that is valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            query_points ((:math:`N_{query\_points}`, 3) :class:`np.ndarray`, optional):
                Query points used to create neighbor pairs. Uses the system's
                points if :code:`None` (Default value = :code:`None`).
            exclude_ii (bool):
                Whether to exclude pairs of particles with the same point index in
                the output neighborlist (Default value = ``True``).
        """
        nq = NeighborQuery.from_system(system)
        box = nq._cpp_obj.getBox()
        points = nq.points
        if query_points is None:
            query_points = points

        points = freud.util._convert_array(points, shape=points.shape, dtype=np.float32)
        query_points = freud.util._convert_array(
            query_points, shape=query_points.shape, dtype=np.float32
        )

        result = cls()
        result._cpp_obj = freud._locality.NeighborList(
            points, query_points, box, exclude_ii
        )

        return result

    def __init__(self, _null=False):
        self._cpp_obj = None if _null else freud._locality.NeighborList()

    def get_ptr(self):
        r"""Returns a pointer to the raw C++ object we are wrapping."""
        return self._cpp_obj

    def copy_c(self, other):
        r"""Copies the contents of other NeighborList into this object."""
        self._cpp_obj.copy(other._cpp_obj)

    def copy(self, other=None):
        r"""Create a copy. If other is given, copy its contents into this
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
        new_copy = NeighborList()
        new_copy.copy(self)
        return new_copy

    def __getitem__(self, key):
        r"""Access the bond array by index or slice."""
        return self._cpp_obj.getNeighbors().toNumpyArray()[key]

    @property
    def query_point_indices(self):
        """(:math:`N_{bonds}`) :class:`np.ndarray`: The query point indices for
        each bond. This array is read-only to prevent breakage of
        :meth:`~.find_first_index()`. Equivalent to indexing with
        :code:`[:, 0]`."""
        return self[:, 0]

    @property
    def point_indices(self):
        """(:math:`N_{bonds}`) :class:`np.ndarray`: The point indices for each
        bond. This array is read-only to prevent breakage of
        :meth:`~.find_first_index()`. Equivalent to indexing with :code:`[:,
        1]`."""
        return self[:, 1]

    @property
    def weights(self):
        """(:math:`N_{bonds}`) :class:`np.ndarray`: The weights for each bond.
        By default, bonds have a weight of 1."""
        return self._cpp_obj.getWeights().toNumpyArray()

    @property
    def distances(self):
        """(:math:`N_{bonds}`) :class:`np.ndarray`: The distances for each
        bond."""
        return self._cpp_obj.getDistances().toNumpyArray()

    @property
    def vectors(self):
        r"""(:math:`N_{bonds}`, 3) :class:`np.ndarray`: The vectors for each
        bond."""
        return self._cpp_obj.getVectors().toNumpyArray()

    @property
    def segments(self):
        """(:math:`N_{query\\_points}`) :class:`np.ndarray`: A segment array
        indicating the first bond index for each query point."""
        return self._cpp_obj.getSegments().toNumpyArray()

    @property
    def neighbor_counts(self):
        """(:math:`N_{query\\_points}`) :class:`np.ndarray`: A neighbor count
        array indicating the number of neighbors for each query point."""
        return self._cpp_obj.getCounts().toNumpyArray()

    def __len__(self):
        r"""Returns the number of bonds stored in this object."""
        return self._cpp_obj.getNumBonds()

    @property
    def num_query_points(self):
        """unsigned int: The number of query points.

        All query point indices are less than this value.
        """
        return self._cpp_obj.getNumQueryPoints()

    @property
    def num_points(self):
        """unsigned int: The number of points.

        All point indices are less than this value.
        """
        return self._cpp_obj.getNumPoints()

    def find_first_index(self, i):
        r"""Returns the lowest bond index corresponding to a query particle
        with an index :math:`\geq i`.

        Args:
            i (unsigned int): The particle index.
        """
        return self._cpp_obj.find_first_index(i)

    def filter(self, filt):
        r"""Removes bonds that satisfy a boolean criterion.

        Args:
            filt (:class:`np.ndarray`):
                Boolean-like array of bonds to keep (True means the bond
                will not be removed).

        .. note:: This method modifies this object in-place.

        Example::

            # Keep only the bonds between particles of type A and type B
            nlist.filter(types[nlist.query_point_indices] != types[nlist.point_indices])
        """  # E501
        filt = np.ascontiguousarray(filt, dtype=bool)
        self._cpp_obj.filter(filt)
        return self

    def filter_r(self, r_max, r_min=0):
        r"""Removes bonds that are outside of a given radius range.

        Args:
            r_max (float):
                Maximum bond distance in the resulting neighbor list.
            r_min (float, optional):
                Minimum bond distance in the resulting neighbor list
                (Default value = :code:`0`).
        """
        self._cpp_obj.filter_r(r_max, r_min)
        return self

    def sort(self, by_distance=False):
        r"""Sort the entries in the neighborlist.

        Args:
            by_distance (bool):
                If ``True``, this method sorts the neighborlist entries by
                ``query_point_index``, then ``distance``, then ``point_index``.
                If ``False``, this method sorts the NeighborList entries by
                ``query_point_index``, then ``point_index``, then ``distance``
                (Default value = ``False``).
        """
        self._cpp_obj.sort(by_distance)
        return self


def _nlist_from_cnlist(c_nlist):
    """Create a Python NeighborList object that points to an existing C++
    NeighborList object.

    This functions generally serves two purposes. Any special locality
    NeighborList generators, like :class:`~.Voronoi`, should use this as a way
    to point to the C++ NeighborList they generate internally. Additionally,
    any compute method that requires a :class:`~.NeighborList` (i.e. cannot do
    with just a :class:`~.NeighborQuery`) should also expose the internally
    computed :class:`~.NeighborList` using this method.

    Args:
        c_nlist (freud._locality.NeighborList):
            C++ neighborlist object.
    """
    result = NeighborList()
    result._cpp_obj = c_nlist
    return result


def _make_default_nq(neighbor_query):
    r"""Helper function to return a NeighborQuery object.

    Currently the resolution for NeighborQuery objects is such that if Python
    users pass in a NumPy array of points and a box, we always make a
    _RawPoints object. On the C++ side, the _RawPoints object internally
    constructs an AABBQuery object to find neighbors if needed. On the Python
    side, making the _RawPoints object is just so that compute functions on the
    C++ side don't require overloads to work.

    Supported types for :code:`neighbor_query` include:
    - :class:`~.locality.AABBQuery`
    - :class:`~.locality.LinkCell`
    - A tuple of :code:`(box, points)` where :code:`box` is a
      :class:`~.box.Box` and :code:`points` is a :class:`numpy.ndarray`.

    Args:
        neighbor_query (:class:`~.locality.NeighborQuery` - like object):
            A :class:`~.locality.NeighborQuery` or object that can be
            duck-typed into one.

    Returns:
        :class:`freud.locality.NeighborQuery`
            The same :class:`NeighborQuery` object if one is given or
            :class:`_RawPoints` built from an inferred :code:`box` and
            :code:`points`.
    """
    if not isinstance(neighbor_query, NeighborQuery):
        nq = _RawPoints(*neighbor_query)
    else:
        nq = neighbor_query
    return nq


def _make_default_nlist(system, neighbors, query_points=None):
    r"""Helper function to return a neighbor list object if is given, or to
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
    """
    if type(neighbors) is NeighborList:
        return neighbors
    query_args = neighbors.copy()
    query_args.setdefault("exclude_ii", query_points is None)
    nq = _make_default_nq(system)
    qp = query_points if query_points is not None else nq.points
    return nq.query(qp, query_args).toNeighborList()


class _RawPoints(NeighborQuery):
    r"""Class containing :class:`~.box.Box` and points with no spatial data
    structures for accelerating neighbor queries."""

    def __init__(self, box, points):
        # Assume valid set of arguments is passed
        b = freud.util._convert_box(box)
        self._points = freud.util._convert_array(
            points, shape=(None, 3), dtype=np.float32
        ).copy()
        self._cpp_obj = freud._locality.RawPoints(b._cpp_obj, self._points)


class AABBQuery(NeighborQuery):
    r"""Use an Axis-Aligned Bounding Box (AABB) tree :cite:`howard2016` to
    find neighbors.

    Also available as ``freud.AABBQuery``.

    Args:
        box (:class:`freud.box.Box`):
            Simulation box.
        points ((:math:`N`, 3) :class:`numpy.ndarray`):
            The points to use to build the tree.
    """

    def __init__(self, box, points):
        # Assume valid set of arguments is passed
        b = freud.util._convert_box(box)
        self._points = freud.util._convert_array(points, shape=(None, 3)).copy()
        self._cpp_obj = freud._locality.AABBQuery(b._cpp_obj, self._points)


class LinkCell(NeighborQuery):
    r"""Supports efficiently finding all points in a set within a certain
    distance from a given point.

    Also available as ``freud.LinkCell``.

    Args:
        box (:class:`freud.box.Box`):
            Simulation box.
        points ((:math:`N`, 3) :class:`numpy.ndarray`):
            The points to bin into the cell list.
        cell_width (float, optional):
            Width of cells. If not provided, :class:`~.LinkCell` will
            estimate a cell width based on the number of points and the box
            size, assuming a constant density of points in the box.
    """

    def __init__(self, box, points, cell_width=0):
        b = freud.util._convert_box(box)
        self._points = freud.util._convert_array(points, shape=(None, 3)).copy()
        self._cpp_obj = freud._locality.LinkCell(b._cpp_obj, self._points, cell_width)

    @property
    def cell_width(self):
        """float: Cell width."""
        return self._cpp_obj.getCellWidth()


class CellQuery(NeighborQuery):
    r"""Use the CellQuery algorithm to find neighbors.

    Also available as ``freud.CellQuery``.

    Args:
        box (:class:`freud.box.Box`):
            Simulation box.
        points ((:math:`N`, 3) :class:`numpy.ndarray`):
            The points to use to build the cell list.
    """

    def __init__(self, box, points):
        # Assume valid set of arguments is passed
        b = freud.util._convert_box(box)
        self._points = freud.util._convert_array(points, shape=(None, 3)).copy()
        self._cpp_obj = freud._locality.CellQuery(b._cpp_obj, self._points)


class _PairCompute(_Compute):
    r"""Parent class for all compute classes in freud that depend on finding
    nearest neighbors.

    The purpose of this class is to consolidate some of the logic for parsing
    the numerous possible inputs to the compute calls of such classes. In
    particular, this class contains a helper function that calls the necessary
    functions to create NeighborQuery and NeighborList classes as needed, as
    well as dealing with boxes and query arguments.
    """

    def _preprocess_arguments(self, system, query_points=None, neighbors=None):
        r"""Process standard compute arguments into freud's internal types by
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
                Query points for preprocessing. Uses :code:`points` if
                :code:`None` (Default value = :code:`None`).
            neighbors (:class:`freud.locality.NeighborList` or :class:`dict`, optional):
                :class:`~.locality.NeighborList` or dictionary of query
                arguments to use to find bonds (Default value = :code:`None`).
        """  # noqa E501
        nq = NeighborQuery.from_system(system)

        # Resolve the two possible ways of passing neighbors (query arguments
        # or neighbor lists) based on the type of the neighbors argument.
        nlist, qargs = self._resolve_neighbors(neighbors, query_points)

        if query_points is None:
            query_points = nq.points
        else:
            query_points = freud.util._convert_array(query_points, shape=(None, 3))
        num_query_points = query_points.shape[0]
        return (nq, nlist, qargs, query_points, num_query_points)

    def _resolve_neighbors(self, neighbors, query_points=None):
        if type(neighbors) is NeighborList:
            nlist = neighbors
            qargs = _QueryArgs()
        elif neighbors is None or type(neighbors) is dict:
            # The default_query_args property must raise a NotImplementedError
            # if no query arguments were passed in and the class has no
            # reasonable choice of defaults.
            try:
                query_args = (
                    self.default_query_args if neighbors is None else neighbors.copy()
                )
                query_args.setdefault("exclude_ii", query_points is None)
                qargs = _QueryArgs.from_dict(query_args)
                nlist = NeighborList(_null=True)
            except NotImplementedError:
                raise
        else:
            msg = (
                "An invalid value was provided for neighbors, "
                "which must be a dict or NeighborList object."
            )
            raise ValueError(msg)
        return nlist, qargs

    @property
    def default_query_args(self):
        """No default query arguments."""
        raise NotImplementedError(
            NO_DEFAULT_QUERY_ARGS_MESSAGE.format(type(self).__name__)
        )


class _SpatialHistogram(_PairCompute):
    r"""Parent class for all compute classes in freud that perform a spatial
    binning of particle bonds by distance.
    """

    def __init__(self):
        # Abstract class
        pass

    @property
    def default_query_args(self):
        """The default query arguments are
        :code:`{'mode': 'ball', 'r_max': self.r_max}`."""
        return dict(mode="ball", r_max=self.r_max)

    @_Compute._computed_property
    def box(self):
        """:class:`freud.box.Box`: The box object used in the last
        computation."""
        return freud.box.BoxFromCPP(self._cpp_obj.getBox())

    @_Compute._computed_property
    def bin_counts(self):
        """:math:`\\left(N_0, N_1 \\right)` :class:`numpy.ndarray`: The bin counts in the histogram."""  # noqa: E501
        return self._cpp_obj.getBinCounts().toNumpyArray()

    @property
    def bin_centers(self):
        """:class:`list` (:class:`numpy.ndarray`): The centers of each bin in the
        histogram (has the same shape in each dimension as the histogram itself)."""
        centers = self._cpp_obj.getBinCenters()
        return [np.array(c) for c in centers]

    @property
    def bin_edges(self):
        """:class:`list` (:class:`numpy.ndarray`): The edges of each bin in the
        histogram (is one element larger in each dimension than the histogram
        because each bin has a lower and upper bound)."""
        edges = self._cpp_obj.getBinEdges()
        return [np.array(e) for e in edges]

    @property
    def bounds(self):
        """:class:`list` (:class:`tuple`): A list of tuples indicating upper and
        lower bounds of each axis of the histogram."""
        return self._cpp_obj.getBounds()  # this returns from cpp with the right type

    @property
    def nbins(self):
        """:class:`list`: The number of bins in each dimension of the
        histogram."""
        return self._cpp_obj.getAxisSizes()

    def _reset(self):
        # Resets the values of SpatialHistogram in memory.
        self._cpp_obj.reset()


class _SpatialHistogram1D(_SpatialHistogram):
    r"""Subclasses _SpatialHistogram to provide a simplified API for
    properties of 1-dimensional histograms.
    """

    def __init__(self):
        # Abstract class
        pass

    @property
    def bin_centers(self):
        """:math:`(N_{bins}, )` :class:`numpy.ndarray`: The centers of each bin
        in the histogram."""
        return np.array(self._cpp_obj.getBinCenters()[0])

    @property
    def bin_edges(self):
        """:math:`(N_{bins}+1, )` :class:`numpy.ndarray`: The edges of each bin
        in the histogram. It is one element larger because each bin has a lower
        and an upper bound."""
        return np.array(self._cpp_obj.getBinEdges()[0])

    @property
    def bounds(self):
        """tuple: A tuple indicating upper and lower bounds of the histogram."""
        return self._cpp_obj.getBounds()[0]

    @property
    def nbins(self):
        """int: The number of bins in the histogram."""
        return self._cpp_obj.getAxisSizes()[0]


class PeriodicBuffer(_Compute):
    r"""Replicate periodic images of points inside a box."""

    def __init__(self):
        self._cpp_obj = freud._locality.PeriodicBuffer()

    def compute(self, system, buffer, images=False, include_input_points=False):
        r"""Compute the periodic buffer.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            buffer (float or list of 3 floats):
                Buffer distance for replication outside the box.
            images (bool, optional):
                If ``False``, ``buffer`` is a distance. If ``True``,
                ``buffer`` is a number of images to replicate in each
                dimension. Note that one image adds half of a box length to
                each side, meaning that one image doubles the box side lengths,
                two images triples the box side lengths, and so on.
                (Default value = :code:`False`).
            include_input_points (bool, optional):
                Whether the original points provided by ``system`` are
                included in the buffer, (Default value = :code:`False`).
        """
        nq = _make_default_nq(system)
        if np.ndim(buffer) == 0:
            # catches more cases than np.isscalar
            buffer_vec = [buffer, buffer, buffer]
        elif len(buffer) == 3:
            buffer_vec = [buffer[0], buffer[1], buffer[2]]
        else:
            msg = "buffer must be a scalar or have length 3."
            raise ValueError(msg)

        self._cpp_obj.compute(nq._cpp_obj, buffer_vec, images, include_input_points)
        return self

    @_Compute._computed_property
    def buffer_points(self):
        """:math:`\\left(N_{buffer}, 3\\right)` :class:`numpy.ndarray`: The
        buffer point positions."""
        return self._cpp_obj.getBufferPoints().toNumpyArray()
        # return np.asarray([[p.x, p.y, p.z] for p in points])

    @_Compute._computed_property
    def buffer_ids(self):
        """:math:`\\left(N_{buffer}\\right)` :class:`numpy.ndarray`: The buffer
        point ids."""
        return self._cpp_obj.getBufferIds().toNumpyArray()

    @_Compute._computed_property
    def buffer_box(self):
        """:class:`freud.box.Box`: The buffer box, expanded to hold the
        replicated points."""
        return freud.box.BoxFromCPP(self._cpp_obj.getBufferBox())

    def __repr__(self):
        return f"freud.locality.{type(self).__name__}()"

    def __str__(self):
        return repr(self)


class Voronoi(_Compute):
    r"""Computes Voronoi diagrams using voro++.

    Voronoi diagrams (`Wikipedia
    <https://en.wikipedia.org/wiki/Voronoi_diagram>`_) are composed of convex
    polytopes (polyhedra in 3D, polygons in 2D) called cells, corresponding to
    each input point. The cells bound a region of Euclidean space for which all
    contained points are closer to a corresponding input point than any other
    input point. A ridge is defined as a boundary between cells, which contains
    points equally close to two or more input points.

    The voro++ library :cite:`Rycroft2009` is used for fast computations of the
    Voronoi diagram.
    """

    def __init__(self):
        self._cpp_obj = freud._locality.Voronoi()

    def compute(self, system):
        r"""Compute Voronoi diagram.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
        """
        nq = NeighborQuery.from_system(system)
        self._cpp_obj.compute(nq._cpp_obj)
        self._box = nq.box
        return self

    @_Compute._computed_property
    def polytopes(self):
        """list[:class:`numpy.ndarray`]: A list of :class:`numpy.ndarray`
        defining Voronoi polytope vertices for each cell."""
        polytopes = []
        raw_polytopes = self._cpp_obj.getPolytopes()
        for i in range(len(raw_polytopes)):
            raw_vertices = raw_polytopes[i]
            num_verts = len(raw_vertices)
            polytope_vertices = np.empty((num_verts, 3), dtype=np.float64)
            for j in range(num_verts):
                polytope_vertices[j, 0] = raw_vertices[j][0]
                polytope_vertices[j, 1] = raw_vertices[j][1]
                polytope_vertices[j, 2] = raw_vertices[j][2]
            polytopes.append(np.asarray(polytope_vertices))
        return polytopes

    @_Compute._computed_property
    def volumes(self):
        """:math:`\\left(N_{points} \\right)` :class:`numpy.ndarray`: Returns
        an array of Voronoi cell volumes (areas in 2D)."""
        return self._cpp_obj.getVolumes().toNumpyArray()

    @_Compute._computed_property
    def nlist(self):
        r"""Returns the computed :class:`~.locality.NeighborList`.

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
        return _nlist_from_cnlist(self._cpp_obj.getNeighborList())

    def __repr__(self):
        return f"freud.locality.{type(self).__name__}()"

    def __str__(self):
        return repr(self)

    def plot(self, ax=None, color_by=None, cmap=None):
        """Plot Voronoi diagram.

        Args:
            ax (:class:`matplotlib.axes.Axes`): Axis to plot on. If
                :code:`None`, make a new figure and axis.
                (Default value = :code:`None`)
            color_by (bool):
                If :code:`'sides'`, color cells by the number of sides.
                If :code:`'area'`, color cells by their area.
                If :code:`None`, random colors are used for each cell.
                (Default value = :code:`None`)
            cmap (str):
                Colormap name to use (Default value = :code:`None`).

        Returns:
            :class:`matplotlib.axes.Axes`: Axis with the plot.
        """
        if not _HAS_MPL:
            raise ImportError(msg_mpl)
        if not self._box.is2D:
            return None
        return freud.plot.voronoi_plot(self, self._box, ax, color_by, cmap)

    def _repr_png_(self):
        try:
            return freud.plot._ax_to_bytes(self.plot())
        except (AttributeError, ImportError):
            return None


class Filter(_PairCompute):
    """Filter an Existing :class:`.NeighborList`.

    This class serves as the base class for all NeighborList filtering methods
    in **freud**. Filtering a :class:`.NeighborList` requires first computing
    the unfiltered :class:`.NeighborList` from a system and a set of query
    arguments. Then, based on the arrangement of particles, their shapes, and
    other criteria determined by the derived class, some of the neighbors are
    removed from the unfiltered :class:`.NeighborList`.

    The compute method of each :class:`.Filter` class will take a system object
    along with a neighbors dictionary specifying query arguments. The
    ``neighbors`` dictionary along with the system object will be used to build
    the unfiltered neighborlist, which will then be filtered according to the
    filter class. After the calculation, the filtered neighborlist will be
    available as the property ``filtered_nlist`` .

    Warning:
        This class is abstract and should not be instantiated directly.
    """

    def __init__(self):
        msg = "The Filter class is abstract and should not be instantiated directly."
        raise RuntimeError(msg)

    def _preprocess_arguments(self, system, query_points=None, neighbors=None):
        """Use a full neighborlist if neighbors=None."""
        nq = NeighborQuery.from_system(system)
        if neighbors is None:
            neighbors = NeighborList.all_pairs(nq, query_points, query_points is None)
        return super()._preprocess_arguments(nq, query_points, neighbors)

    def compute(self, system, neighbors=None, query_points=None):
        r"""Filter a :class:`.Neighborlist`.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                Either a :class:`NeighborList` of neighbor pairs to use for the
                unfiltered neighbor list, or a dictionary of `query arguments
                <https://freud.readthedocs.io/en/stable/topics/querying.html>`__.
                If ``None``, an unfiltered neighborlist will be created such that
                all pairs of particles are neighbors via :meth:`.NeighborList.all_pairs`
                (Default value = ``None``).
            query_points ((:math:`N_{query\_points}`, 3) :class:`np.ndarray`, optional):
                Query points used to calculate the unfiltered neighborlist. Uses
                the system's points if :code:`None` (Default value = :code:`None`).
        """
        nq, nlist, qargs, query_points, _num_query_points = self._preprocess_arguments(
            system, query_points, neighbors
        )

        self._cpp_obj.compute(nq._cpp_obj, query_points, nlist._cpp_obj, qargs._cpp_obj)
        return self

    @_Compute._computed_property
    def filtered_nlist(self):
        """:class:`.NeighborList`: The filtered neighbor list."""
        return _nlist_from_cnlist(self._cpp_obj.getFilteredNlist())

    @_Compute._computed_property
    def unfiltered_nlist(self):
        """:class:`.NeighborList`: The unfiltered neighbor list."""
        return _nlist_from_cnlist(self._cpp_obj.getUnfilteredNlist())


class FilterSANN(Filter):
    """Filter a :class:`.NeighborList` via the SANN method.

    The Solid Angle Nearest Neighbor (SANN) method :cite:`vanMeel2012` is a
    parameter-free algorithm for the identification of nearest neighbors. The
    SANN method attributes to each possible neighbor a solid angle and determines
    the cutoff radius by the requirement that the sum of the solid angles is 4π.

    For performance considerations, SANN is implemented as a way of filtering
    a pre-existing set of neighbors due to the high performance cost of sorting
    all :math:`N^2` particle pairs by distance. For a more in-depth explanation of
    the neighborlist filter concept in **freud**, see :class:`.Filter`.

    Warning:
        Due to the above design decision, it is possible that the unfiltered
        neighborlist will not contain enough neighbors to completely fill the
        neighbor shell of some particles in the system. The ``allow_incomplete_shell``
        argument to :class:`.FilterSANN`'s constructor controls whether a warning
        or exception is raised in these cases.

    Note:
        The ``filtered_nlist`` computed by this class will be sorted by distance.

    Note:
        We recommend using unfiltered neighborlists in which no particles are their
        own neighbor.

    Args:
        allow_incomplete_shell (bool):
            Whether particles with incomplete neighbor shells are allowed in the
            filtered neighborlist. If True, a warning will be raised if there are
            particles with incomplete neighbors shells in the filtered neighborlist.
            If False, an exception will be raised in the same case (Default value =
            :code:`False`).
    """

    def __init__(self, allow_incomplete_shell=False):
        self._cpp_obj = freud._locality.FilterSANN(allow_incomplete_shell)


class FilterRAD(Filter):
    """Filter a :class:`.NeighborList` via the RAD method.

    The Relative Angular Distance (RAD) method :cite:`Higham2016` is a parameter-free
    algorithm for the identification of nearest neighbors. A particle's neighbor shell
    is taken to be all particles that are not blocked by any other particle.

    The :class:`.FilterRAD` algorithm considers the potential neighbors of a query point
    :math:`i` going radially outward, and filters the neighbors :math:`j` of :math:`i`
    which are blocked by a closer neighbor :math:`k`. The RAD algorithm may filter
    out all further neighbors of :math:`i` as soon as blocked neighbor :math:`j` is
    found. This is the mode corresponding to ``terminate_after_blocked=True`` and is
    called "RAD-closed" in :cite:`Higham2016`. If ``terminate_after_blocked=False``,
    then :class:`.FilterRAD` will continue to consider neighbors further away than
    :math:`j`, only filtering them if they are blocked by a closer neighbor. This mode
    is called "RAD-open" in :cite:`Higham2016`.

    RAD is implemented as a filter for pre-existing sets of neighbors due to
    the high performance cost of sorting all :math:`N^2` particle pairs by
    distance. For a more in-depth explanation of the neighborlist filter
    concept in **freud**, see :class:`.Filter`.

    Warning:
        Due to the above design decision, it is possible that the unfiltered
        neighborlist will not contain enough neighbors to completely fill the
        neighbor shell of some particles in the system. The ``allow_incomplete_shell``
        argument to :class:`.FilterRAD`'s constructor controls whether a warning
        or exception is raised in these cases.

    Note:
        The ``filtered_nlist`` computed by this class will be sorted by distance.

    Note:
        We recommend using unfiltered neighborlists in which no particles are their
        own neighbor.

    Args:
        allow_incomplete_shell (bool):
            Whether particles with incomplete neighbor shells are allowed in the
            filtered neighborlist. If True, a warning will be raised if there are
            particles with incomplete neighbors shells in the filtered neighborlist.
            If False, an exception will be raised in the same case. Only considered
            when ``terminate_after_blocked=True`` (Default value = :code:`False`).
        terminate_after_blocked (bool):
            Filter potential neighbors after a closer blocked particle is found
            (Default value = :code:`False`).
    """

    def __init__(self, allow_incomplete_shell=False, terminate_after_blocked=True):
        self._cpp_obj = freud._locality.FilterRAD(
            allow_incomplete_shell, terminate_after_blocked
        )
