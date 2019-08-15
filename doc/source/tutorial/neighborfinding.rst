.. _neighbors:

================
Neighbor Finding
================

Problem Statement
=================

A central task in many of the computations in **freud** is finding particles' neighbors.
Many of the methods characterize the environments of particles based on their relative positions.
As a result, the computations involve performing pair calculations.
Although in principle such calculations should typically be conducted for all possible pairs, it is usually sufficient to perform analyses only on particles within a given distance that characterizes the local environment.
This requirement is analogous to the force calculations typically performed in molecular dynamics simulations, where a cutoff radius is specified beyond which pair forces are assumed to be small enough to neglect.


Neighbor Querying
=================

To support efficiently finding neighbors in periodic systems, the **freud** library provides a unified interface to finding neighbors that can both be accessed directly by users and is leveraged internally by all methods that require such calculations.
The central interface for neighbor finding is the :py:class:`freud.locality.NeighborQuery` family of classes, which provide methods for dynamically finding neighbors given a :py:class:`freud.box.Box`.
The :py:class:`freud.locality.NeighborQuery` class defines an abstract interface for neighbor finding that is implemented by its subclasses, namely the :py:class:`freud.locality.LinkCell` and :py:class:`freud.locality.AABBQuery` classes.
These classes represent data structures used to accelerate neighbor finding.

In general, these data structures operate by constructing them using one set of points, after which they can be queried to efficiently find the neighbors of arbitrary other points.
The queries can either be based strictly on a distance cutoff (using :py:class:`freud.locality.NeighborQuery.queryBall`), or by searching for a specific number of neighbors (using :py:class:`freud.locality.NeighborQuery.query`).
These classes expose a standardized API for this task, returning a generator than can be easily looped over.
Alternatively, the pairs of points can be converted into a :py:class:`freud.locality.NeighborList`, a lightweight class that can be used to store pairs of bonds that will be used multiple times.
For example, the following code first uses the :py:class:`freud.locality.LinkCell` to search for at least 6 neighbors for all points in a given set and calculate the average distances between them, and then uses a :py:class:`freud.locality.AABBQuery` to create a :py:class:`freud.locality.NeighborList` based on a distance cutoff:

.. code-block:: python

    import numpy as np
    import freud

    # As an example, we randomly generate 100 points in a 10x10x10 cubic box.
    L = 10
    num_points = 100

    # Note that we are shifting all points into the expected range for freud.
    # An alternative would be to simply wrap all points back into the box using
    # the box.wrap method.
    points = np.random.rand(num_points)*L - L/2
    box = freud.box.Box.cube(L)

    # Linked cell lists are parameterized by the size of the individual cells.
    # The cell width significantly affects neighbor-finding performance.
    # In general, for a single distance-based query a good choice for the cell
    # width is the actual distance to query; in this case, since we are asking
    # for a specific number of neighbors, we simply provide a reasonable guess.
    cell_width = 2
    lc = freud.locality.LinkCell(box=box, points=points, cell_width=cell_width)
    distances = []
    for bond in lc.query(query_points=points, num_neighbors=4):
        # The returned bonds are tuples of the form
        # (index_query_point, index_point, distance).
        distances.append(bond[2])
    avg_distance = np.mean(distances)

    # The result of a query object can be transparently converted into a
    # NeighborList object for future use in computations. In this example,
    # we instead perform a distance-based query using the queryBall method.
    query_distance = 3
    freud.locality.AABBQuery(box=box, points=points).queryBall(
        query_points=points, r=query_distance).toNeighborList()


Neighbor Computations
=====================

Classes that actually involve finding neighbors in general expose an API that maps directly onto the neighbor querying API.
Like all compute classes, they expose a ``compute`` method, but in general these methods have a signature ``compute(nq, query_points=None, query_args={}, nlist=None, ...)`` (potentially with additional arguments as signified by the ellipsis).
The ``nq`` argument can be either a :py:class:`freud.locality.NeighborQuery` or a tuple ``(box, points)``, where ``box`` and ``points`` have the usual meanings as elsewhere in **freud**.

The API for these classes is intended to offer maximal flexibility, allowing users to choose the fastest way to iterate over neighbors for a given computation, but the options can be a bit complex, so we provide an overview here.
The simplest usage of one of these classes is to simply call ``compute((box, points), METHOD_SPECIFIC_ARGS...)``, in which case the class will internally build a :py:class:`freud.locality.NeighborQuery` class using the points and then find their neighbors.
However, if the user expects to perform multiple different **freud** calculations (for instance, the calculation of various order parameters) on the same pairs of points, it is worthwhile to cache the :py:class:`freud.locality.NeighborQuery` object between calls and pass it into each of the computations to spare the cost of rebuilding the object.

.. code-block:: python

    # For simplicity, let's reuse the configuration used above.
    L = 10
    num_points = 100
    points = np.random.rand(num_points)*L - L/2
    box = freud.box.Box.cube(L)

    # First, let's compute an RDF using the straightforward API.
    rdf = freud.density.RDF(rmax=5, dr=0.1).compute(
        (box, points))

    # Now, let's instead reuse the object for a pair of calculations:
    nq = freud.locality.AABBQuery(box=box, points=points)
    rdf = freud.density.RDF(rmax=5, dr=0.1).compute(nq)

    nbins = 100
    rmax = 4
    orientations = np.array([[1, 0, 0, 0]*num_points)
    pmft = freud.pmft.PMFTXYZ(rmax, rmax, rmax, nbins, nbins, nbins)
    pmft.compute(nq, orientations=orientations)


If the user in fact expects to perform computations with the exact same pairs of neighbors (for example, to compute :py:class:`freud.order.Steinhardt` for multiple :math:`l` values), then the user can further speed up the calculation by precomputing the entire :py:class:`freud.locality.NeighborList` and storing it for future use.
In this case, if the user passes in ``compute(nq=(box, points))``, **freud** will not spend the time to construct a :py:class:`freud.locality.NeighborQuery`, knowing that no querying is necessary because a :py:class:`freud.locality.NeighborList` has been provided.
This mode of operation is particularly useful when the user wishes to use a :py:class:`freud.locality.NeighborList` computed using an alternate method, such as a Voronoi cell calculation.


.. code-block:: python

    # Reusing the AABBQuery object constructed in the previous example,
    # we first attempt the computation for various values of l using a
    # distance based cutoff for neighbor finding.
    rmax = 3
    nlist_ball = nq.queryBall(points, r=rmax)
    q6_ball_arrays = []
    for l in range(3, 6):
        ql = freud.density.Steinhardt(l=l)
        q6_ball_arrays.append(ql.compute((box, points), nlist_ball).order)

    nlist_number = nq.queryBall(points, num_neighbors=6)
    q6_number_arrays = []
    for l in [4, 6, 8]:
        q6 = freud.density.Steinhardt(l=l)
        q6_number_arrays.append(ql.compute((box, points), nlist_number).order)


Notably, in this example we used two different methods for finding neighbors.
Just as the querying interface for :py:class:`freud.locality.NeighborQuery` classes offers both of these approaches, all neighbor compute classes also offer this flexibility through the ``query_args`` argument.
This argument accepts a dictionary of inputs that will be passed through to the underlying query call, allowing the user flexibility as to how neighbors are defined.
For instance, to replicate the above results, we could instead do the following:


.. code-block:: python

    # Reusing the AABBQuery object constructed in the previous example,
    # we first attempt the computation for various values of l using a
    # distance based cutoff for neighbor finding.
    q6_ball_arrays = []
    for l in range(3, 6):
        ql = freud.density.Steinhardt(l=l)
        q6_ball_arrays.append(
            ql.compute((box, points),
                       query_args={'mode': 'ball', 'r': rmax}).order)

    nlist_number = nq.queryBall(points, num_neighbors=6)
    for l in range(3, 6):
        q6 = freud.density.Steinhardt(l=l)
        q6_number_arrays.append(ql.compute((box, points), nlist_order).order)
        q6_number_arrays.append(
            ql.compute((box, points),
                       query_args={'mode': 'nearest', 'nn': 6}).order)

Since most computations in **freud** have sensible defaults, the query arguments for a given class will be populated automatically if not provided by the user.
The defaults for each class are given in the class's documentation.
In general, the valid keys for the ``query_args`` dictionary are ``mode``,  ``r``,  ``nn``, and ``exclude_ii``.
We have seen examples of the first three above; to understand the last one, we must consider slightly more complex systems, namely binary systems.


Binary Computations
+++++++++++++++++++

The final argument of note is the ``query_points`` argument.
In some cases, it may be useful to perform a calculation with two distinct sets of points.
For instance, in a binary system we might be interested in the :py:class:`freud.pmft.PMFTXYZ` of one particle type with respect to the other.
In such cases, we can use the ``query_points`` argument to perform this calculation:

.. code-block:: python

    pmft = freud.pmft.PMFTXYZ(rmax, rmax, rmax, nbins, nbins, nbins)
    positions_A = ...
    positions_B = ...

    # For simplicity, assume all particles are isotropic (spherical) and
    # therefore orientations are irrelevant.
    orientations = np.array([[1, 0, 0, 0]*num_points)

    pmft.compute((box, positions_A), orientations=orientations,
                 query_points=positions_B, query_orientations=orientations)

Let's take a moment to understand exactly what this calculation means.
Internally, **freud** will create a :py:class:`freud.locality.NeighborQuery` object using ``(box, positions_A)``, and then call its ``queryBall`` method, passing in ``positions_B`` and ``rmax`` as the query arguments (default for PMFTs that can, of course, be overridden using the ``query_args``).
The resulting sets of neighbors will then be used in the calculation of the PMFT.
As a result, we will calculate the PMFT of the distribution of ``positions_A`` around centers located at ``positions_B``.

Understanding this sequence is particularly important when using the ``nearest`` mode of neighbor finding, because *this mode is not symmetric*.
To understand what this means, consider the following simple example:


.. code-block:: python

    positions_A = [[0, 0, 0]]
    positions_B = [[-1, 0, 0], [1, 0, 0]]
    nq_A = freud.locality.AABBQuery(box=box, points=positions_A)
    count_A = 0
    for _ in nq_A.query(positions_B, num_neighbors=1):
        count_A += 1

    nq_B = freud.locality.AABBQuery(box=box, points=positions_B)
    count_B = 0
    for _ in nq_B.query(positions_A, num_neighbors=1):
        count_B += 1

    print(count_A)
    >>> 2

    print(count_B)
    >>> 1

The reason these calculations give different results is simple.
We only asked for one neighbor for each point, but in the first case we queried for two points, and as a result, it found us one neighbor for each of the two points.
In the second case, we constructed our object with two points, but then only requested the neighbors for the single point in ``positions_A``; as a result, we only found one neighbor.
This logic is precisely what governs the ``query_points`` argument; the ``points`` are used to build the query object, and the ``query_points`` are what is passed to the query methods.

Neighbor Self-Exclusion
+++++++++++++++++++++++

We are now in a position to explain the ``exclude_ii`` query argument.
When performing a calculation on a single-component system, we typically do not wish to include a particle as its own neighbor.
The ``exclude_ii`` argument indicates to a query method that any pair of particles with identical indices in the ``points`` and ``query_points`` arrays should be ignored.
Since this behavior is generally expected for single-component systems, it is automatically set to ``True`` in compute classes if ``query_points`` are not passed in explicitly (in which case the ``points`` are reused as the ``query_points``).
Conversely, the argument defaults to ``False`` when ``query_points`` are explicitly provided.
In both cases, the user can override the default behavior by passing the argument explicitly in ``query_args``.
