.. _optimizing:

===========================
Using **freud** Efficiently
===========================

The **freud** library is designed to be both fast and easy-to-use.
In many cases, the library's performance is good enough that users don't need to worry about their usage patterns.
However, in highly performance-critical applications (such as real-time visualization or on-the-fly calculations mid-simulation), uses can benefit from knowing the best ways to make use of **freud**.
This page provides some guidance on this topic.

Reusing Locality Information
============================

Perhaps the most powerful method users have at their disposal for speeding up calculations is proper reuse of the data structures in :mod:`freud.locality`.
As one example, consider using **freud** to calculate multiple neighbor-based quantities for the same set of data points.
It is important to recognize that internally, each time such a calculation is performed using a ``(box, points)`` :class:`tuple`, the compute class is internally rebuilding a neighbor-finding accelerator such a :class:`freud.locality.AABBQuery` object and then using it to find neighbors:

.. code-block:: python

    # Behind the scenes, freud is essentially running
    # freud.locality.AABBQuery(box, points).query(points, dict(r_max=5, exclude_ii=True))
    # and feeding the result to the RDF calculation.
    rdf = freud.density.RDF(bins=50, r_max=5)
    rdf.compute(system=(box, points))


If users anticipate performing many such calculations on the same system of points, they can amortize the cost of rebuilding the :class:`AABBQuery <freud.locality.AABBQuery>` object by constructing it once and then passing it into multiple computations:

.. code-block:: python

    # Now, let's instead reuse the object for a pair of calculations:
    nq = freud.locality.AABBQuery(box=box, points=points)
    rdf = freud.density.RDF(bins=50, r_max=5)
    rdf.compute(system=nq)

    r_max = 4
    orientations = np.array([[1, 0, 0, 0]] * num_points)
    pmft = freud.pmft.PMFTXYZ(r_max, r_max, r_max, bins=100)
    pmft.compute(system=nq, orientations=orientations)

This reuse can significantly improve performance in e.g. visualization contexts where users may wish to calculate a :class:`bond order diagram <freud.environment.BondOrder>` and an :class:`RDF <freud.density.RDF>` at each frame, perhaps for integration with a visualization toolkit like `OVITO <https://www.ovito.org/>`_.

A slightly different use-case would be the calculation of multiple quantities based on *exactly the same set of neighbors*.
If the user in fact expects to perform computations with the exact same pairs of neighbors (for example, to compute :class:`freud.order.Steinhardt` for multiple :math:`l` values), then the user can further speed up the calculation by precomputing the entire :class:`freud.NeighborList` and storing it for future use.

.. code-block:: python

    r_max = 3
    nq = freud.locality.AABBQuery(box=box, points=points)
    nlist = nq.query(points, dict(r_max=r_max)).toNeighborList()
    q6_arrays = []
    for l in range(3, 6):
        ql = freud.order.Steinhardt(l=l)
        q6_arrays.append(ql.compute((box, points), neighbors=nlist).particle_order)


Notably, if the user calls a compute method with ``compute(system=(box, points))``, unlike in the examples above **freud** **will not construct** a :class:`freud.locality.NeighborQuery` internally because the full set of neighbors is completely specified by the :class:`NeighborList <freud.NeighborList>`.
In all these cases, **freud** does the minimal work possible to find neighbors, so judicious use of these data structures can substantially accelerate your code.

Proper Data Inputs
==================

Minor speedups may also be gained from passing properly structured data to **freud**.
The package was originally designed for analyzing particle simulation trajectories, which are typically stored in single-precision binary formats.
As a result, the **freud** library also operates in single precision and therefore converts all inputs to single-precision.
However, NumPy will typically work in double precision by default, so depending on how data is streamed to **freud**, the package may be performing numerous data copies in order to ensure that all its data is in single-precision.
To avoid this problem, make sure to specify the appropriate data types (:attr:`numpy.float32`) when constructing your NumPy arrays.
