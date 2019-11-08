================
Neighbor Finding
================

Neighbor finding is central to many methods in **freud**.
The purpose of this page is to introduce the various C++ classes and functions involved in the neighbor finding process.
This page focuses on aspects of the neighbor finding utilities in **freud** that are important for developers.
As such, it assumes knowledge on the level of :ref:`neighbors` and :ref:`paircompute`, so please familiarize yourself with the contents of those pages before proceeding.

There are two primary use-cases for the neighbor finding code in **freud**.
One is to directly expose this functionality to the user, via the :class:`~freud.locality.NeighborQuery` abstract class and its subclasses.
The second it to enable looping over nearest neighbors (as defined by arbitrary query arguments or a precomputed :class:`~freud.locality.NeighborList`) inside of compute methods defined in C++.
To support both of these use-cases, **freud** defines how to find neighbors inside iterator classes, which can be naturally looped over in either case.
In this page, we first discuss these iterators and how they are structured with respect to the ``locality::NeighborQuery`` C++ class.
We then discuss the utility functions built around this class to enable easier C++ computations, at which point we also discuss how :class:`~freud.locality.NeighborList` objects fit into this framework.

Per-Point Iterators and the NeighborQuery class
===============================================

The lowest-level unit of the neighbor finding infrastructure in **freud** is the ``locality::NeighborPerPointIterator``.
This class defines an abstract interface for all neighbor iteration in **freud**, an interface essentially composed of the ``next`` and ``end`` methods.
Given an instance of this class, these two methods provide the means for client code to get the next neighbor in the iteration until there are no further neighbors.
Calls to ``next`` produces instances of ``locality::NeighborBond``, a simple data class that contains the core information defining a bond (a pair of points, a distance, and any useful ancillary information).

The rationale for making per-point iteration the central element is twofold.
The first is conceptual: all logic for finding neighbors is naturally reducible to a set of conditions on the neighbors of each query point.
The second is more practical: since finding neighbors for each point must be sequential in many cases (such as nearest neighbor queries), per-point iteration is the smallest logical unit that can be parallelized.

Instances of ``locality::NeighborPerPointIterator`` should not be constructed directly; instead, they should be constructed via the ``locality::NeighborQuery::querySingle`` method.
The ``locality::NeighborQuery`` class is an abstract data type that defines an interface for any method implemented for finding neighbors.
All subclasses must implement the ``querySingle`` method, which should return a subclass of ``locality::NeighborPerPointIterator``.
For instance, the ``locality::AABBQuery`` class implements ``querySingle``, which returns an instance of the ``locality::AABBIterator`` subclass.
In general, different ``NeighborQuery`` subclasses will need to implement separate per-point iterators for each query mode; for ``locality::AABBQuery``, these are the ``locality::AABBQueryIterator`` and the ``locality::AABBQueryBallIterator``, which encode the logic for nearest neighbor and ball queries, respectively.

Although the ``querySingle`` method is what subclasses should implement, the primary interface to ``NeighborQuery`` subclasses is the ``query`` method, which accepts an arbitrary set of query points and query arguments and simply generates a ``locality::NeighborQueryIterator`` object.
The ``locality::NeighborQueryIterator`` class is an intermediary that allows lazy generation of neighbors.
It essentially functions as the container for a set of points, query points, and query arguments; once iteration of this object begins, it produces ``NeighborPerPointIterator`` objects on demand.
This mode of operation also enables the generator approach to looping over neighbors in Python, since iterating in Python corresponds directly to calling ``next`` on the underlying ``NeighborQueryIterator``.

There is one conceptual complexity associated with this class that is important to understand.
Since all of the logic for finding neighbors is contained in the per-point iterator classes, the ``NeighborQueryIterator`` actually retains a reference to the constructing ``NeighborQuery`` object so that it can call ``querySingle`` for each point.
This bidirectionally linked structure enables encapsulation of the neighbor finding logic while also supporting easily parallelization (via parallel calls to ``querySingle``).
Additionally, this structure makes it natural to generate ``NeighborList`` objects.

NeighborLists
=============

The ``NeighborList`` class represents a static list of neighbor pairs.
The results of any query can be converted to a ``NeighborList`` by calling the ``toNeighborList`` method of the ``NeighborQueryIterator``, another reason why the iterator class logic is separated from the ``NeighborQuery`` object: it allows generation of a ``NeighborList`` from -- and more generally, independent operation on -- the result of a query.
The ``NeighborList`` is simply implemented as a collection of raw arrays, one of which holds pairs of neighbors.
The others hold any additional information associated with each bond, such as distances or weights.

By definition, the bonds in a ``NeighborList`` are stored in the form ``(query_point, point)`` (i.e. this is how the underlying array is indexed) and ordered by ``query_point``.
This ordering makes the structure amenable to a fast binary search algorithm.
Looping over neighbors of a given ``query_point`` is then simply a matter of finding the first index in the list where that ``query_point`` appears and then iterating until the ``query_point`` index in the list no longer matches the one under consideration.

Computing With Neighbors
========================

One of the most common operations in **freud** is performing some computation over all neighbor-bonds in the system.
Users have multiple ways of specifying neighbors (using query arguments or by a :class:`~freud.locality.NeighborList`), so **freud** provides some utility functions to abstract the process of looping over neighbors.
These functions are defined in ``locality/NeighborComputeFunctional.h``; the two most important ones are ``loopOverNeighbors`` and ``loopOverNeighborsIterator``.
Compute functions that perform neighbor computations typically accept a ``NeighborQuery``, a ``QueryArgs``, and a ``NeighborList`` object.
These objects can then be passed to either of the utility functions, which loop over the ``NeighborList`` if it was provided (if no :class:`~freud.locality.NeighborList` is provided by the Python user, a ``NULL`` pointer is passed through), and if not, perform a query on the ``NeighborQuery`` object using the provided ``QueryArgs`` to generate the required neighbors.
The actual computation should be encapsulated as a lambda function that is passed as an argument to these utilities.

The distinction between the two utility functions lies in the signature of the accepted lambda functions, which enables a slightly different form of computation.
The default ``loopOverNeighbors`` function does exactly what is described above, namely it calls the provided compute function for every single bond.
However, some computations require some additional code to be executed for each ``query_point``, such as some sort of normalization.
To enable this mode of operation, the ``loopOverNeighborsIterator`` method instead requires a lambda function that accepts two arguments, the ``query_point`` index and a ``NeighborPerPointIterator``.
This way, the client code can loop over the neighbors of a given ``query_point`` and perform the needed computation, then execute additional code (which may optionally depend on the index of the ``query_point``, e.g. to update a specific array index).

Default Systems
---------------

There is one important implementation detail to note.
The user is permitted to simply provide a set of points rather than a ``NeighborQuery`` object on the Python side (i.e. any valid argument to :meth:`~freud.locality.NeighborQuery.from_system`), but we need a natural way to mirror this in C++, ideally without too many method overloads.
To implement this, we provide the ``RawPoints`` C++ class and its Python :class:`~freud.locality._RawPoints` mirror, which is essentially a plain container for a box and a set of query points.
This object inherits from ``NeighborQuery``, allowing it to be passed directly into the C++ compute methods.

However, neighbor computations still need to know how to find neighbors.
In this case, they must construct a ``NeighborQuery`` object capable of neighbor finding and then use the provided query arguments to find neighbors.
To enable this calculation, the ``RawPoints`` class implements a query method that simply constructs an ``AABBQuery`` internally and queries it for neighbors.

Default NeighborLists
---------------------

Some compute methods are actually computations that produce quantities per bond.
One example is the ``SolidLiquid`` order parameter, which computes an order parameter value for each bond.
The ``NeighborComputeFunctional.h`` file implements a ``makeDefaultNList`` function that supports this calculation by creating a ``NeighborList`` object from whatever inputs are provided on demand.
