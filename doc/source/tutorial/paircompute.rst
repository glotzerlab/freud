.. _paircompute:

=================
Pair Computations
=================

Some computations in **freud** do not depend on bonds at all.
For example, :class:`freud.density.GaussianDensity` creates a "continuous equivalent" of a system of points by placing normal distributions at each point's location to smear out its position, then summing the value of these distributions at a set of fixed grid points.
This calculation can be quite useful because it allows the application of various analysis tools like fast Fourier transforms, which require regular grids.
For the purposes of this tutorial, however, the importance of this class is that it is an example of a calculation where neighbors are unimportant: the calculation is performed on a per-point basis only.

The much more common pattern in **freud**, though, is that calculations involve the local neighborhoods of points.
To support efficient, flexible computations of such quantities, various ``Compute classes`` essentially expose the same API as the query interface demonstrated in the previous section.
These ``PairCompute classes`` are designed to mirror the querying functionality of **freud** as closely as possible.

As an example, let's consider :class:`freud.density.LocalDensity`, which calculates the density of points in the local neighborhood of each point.
Adapting our code from the `previous section <neighborfinding>`, the simplest usage of this class would be as follows:

.. code-block:: python

    import numpy as np
    import freud

    L = 10
    num_points = 100

    points = np.random.rand(num_points)*L - L/2
    box = freud.box.Box.cube(L)

    # r_max specifies how far to search around each point for neighbors
    r_max = 2

    # For systems where the points represent, for instance, particles with a
    # specific size, the diameter is used to add fractional volumes for
    # neighbors that would be overlapping the sphere of radius r_max around
    # each point.
    diameter = 0.001

    ld = freud.density.LocalDensity(r_max, diameter)
    ld.compute((box, points))

    # Access the density.
    ld.density

Using the same example system we've been working with so far, we've now calculated an estimate for the number of points in the neighborhood of each point.
Since we already told the computation how far to search for neighbors based on ``r_max``, all we had to do was pass a :class:`tuple` ``(box, points)`` to compute indicate where the points were located.

Binary Systems
==============

Imagine that instead of a single set of points, we actually had two different types of points and we were interested in finding the density of one set of points in the vicinity of the other.
In that case, we could modify the above calculation as follows:


.. code-block:: python

    import numpy as np
    import freud
    L = 10
    num_points = 100
    points = np.random.rand(num_points)*L - L/2
    query_points = np.random.rand(num_points/10)*L - L/2

    r_max = 2
    diameter = 0.001

    ld = freud.density.LocalDensity(r_max, diameter)
    ld.compute((box, points), query_points)

    # Access the density.
    ld.density

The choice of names names here is suggestive of exactly what this calculation is now doing.
Internally, :class:`freud.density.LocalDensity` will search for all ``points`` that are within the cutoff distance ``r_max`` of every ``query_point`` (essentially using the query interface we introduced previously) and use that to calculate ``ld.density``.
Note that this means that ``ld.density`` now contains densities for every ``query_point``, i.e. it is of length 10, not 100.
Moreover, recall that one of the features of the querying API is the specification of whether or not to count particles as their own neighbors.
``PairCompute classes`` will attempt to make an intelligent determination of this for you; if you do not pass in a second set of ``query_points``, they will assume that you are computing with a single set of points and automatically exclude self-neighbors, but otherwise all neighbors will be included.

So far, we have included all points within a fixed radius; however, one might instead wish to consider the density in some shell, such as the density between 1 and 2 distance units away.
To address this need, you could simply adapt the call to ``compute`` above as follows:

.. code-block:: python

    ld.compute((box, points), query_points,
               neighbors=dict(r_max=2, r_min=1))

The ``neighbors`` argument to ``PairCompute`` classes allows users to specify arbitary query arguments, making it possible to easily modify **freud** calculations on-the-fly.
The ``neighbors`` argument is actually more general than query arguments you've seen so far: if query arguments are not precise enough to specify the exact set of neighbors you want to compute with, you can instead provide a :class:`NeighborList <freud.locality.NeighborList>` directly

.. code-block:: python

    ld.compute(neighbor_query=(box, points), query_points=query_points,
               neighbors=nlist)

This feature allows users essentially arbitrary flexibility to specify the bonds that should be included in any bond-based computation.
A common use-case for this is constructing a :class:`NeighborList <freud.locality.NeighborList>` using :class:`freud.locality.Voronoi`; Voronoi constructions provide a powerful alternative method of defining neighbor relationships that can improve the accuracy and robustness of certain calculations in **freud**.

You may have noticed in the last example that all the arguments are specified using keyword arguments.
As the previous examples have attempted to show, the ``query_points`` argument defines a second set of points to be used when performing calculations on binary systems, while the ``neighbors`` argument is how users can specify which neighbors to consider in the calculation.

The ``neighbor_query`` argument is what, to this point, we have been specifying as a :class:`tuple` ``(box, points)``.
As the name suggests, however, we don't have to use this tuple.
Instead, we can pass in any :class:`freud.locality.NeighborQuery`, the central class in **freud**'s querying infrastructure.
In fact, you've already seen examples of :class:`freud.locality.NeighborQuery`: the :class:`freud.locality.AABBQuery` object that we originally used to find neighbors.
Since these objects all contain a :class:`freud.box.Box` and a set of points, they can be directly passed to computations:

.. code-block:: python

    aq = freud.locality.AABBQuery(box, points)
    ld.compute(aq, query_points, nlist)

For more information on why you might want to use :class:`freud.locality.NeighborQuery` objects instead of the tuples, see `optimizing`_.
For now, just consider this to be a way in which you can simplify your calls to many **freud** computes in one script by storing ``(box, points)`` into another objects.

You've now covered the most important information needed to use **freud**!
To recap, we've discussed how **freud** handles periodic boundary conditions, the structure and usage of ``Compute classes``, and methods for finding and performing calculations with pairs of neighbors.
For more detailed information on specific methods in **freud**, see the `examples`_ page or look at the API documentation for specific modules.
