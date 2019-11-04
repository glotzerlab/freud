==========
Histograms
==========

Histograms are a common type of calculation implemented in **freud** because custom histograms are hard to compute efficiently in pure Python.
The C++ ``Histogram`` class support weighted N-dimensional histograms with different spacings in each dimension.
The key to this flexibility is the ``Axis`` class, which defines the range spacing along a single axis; an N-dimensional ``Histogram`` is composed of a sequence of N ``Axis`` objects.
Binning values into the histogram is performed by binning along each axis.
The standard ``RegularAxis`` subclass of ``Axis`` defines an evenly spaced axis with bin centers defined as the center of each bin; additional subclasses may be easily defined to add different spacing if desired.

Multithreading is achieved through the ``ThreadLocalHistogram`` class, which is a simple wrapper around the ``Histogram`` that creates an equivalent histogram on each thread.
The standard pattern for parallel histogramming is to generate a ``ThreadLocalHistogram`` and add data into it, then call the ``Histogram::reduceOverThreads`` method to accumulate these data into a single histogram.
In case any additional post-processing is required per bin, it can also be executed in parallel by providing it as a lambda function to ``Histogram::reduceOverThreadsPerBin``.


Computing with Histograms
=========================

The ``Histogram`` class is designed as a data structure for the histogram.
Most histogram computations in **freud** involve standard neighbor finding to get bonds, followed by binning some function of these bonds into a histogram.
Examples include RDFs (binning bond distances), PMFTs (binning bonds by the different vector components of the bond), and bond order diagrams (binning bond angles).
An important distinction between histogram computations and most others is that histograms naturally support an accumulation of information over multiple frames of data, an operation that is ill-defined for many other computations.
As a result, histogram computations also need to implement some boilerplate for handling accumulating and averaging data over multiple frames.

The details of these computations are encapsulated by the ``BondComputeHistogram`` class, which contains a histogram, provides accessors to standard histogram properties like bin counts and axis sizes, and has a generic accumulation method that accepts a lambda compute function.
This signature is very similar to the utility functions for looping over neighbors, and in fact the function is transparently forwarded to ``locality::loopOverNeighbors``.
Any compute that matches this pattern should inherit from the ``BondComputeHistogram`` class and must implement an ``accumulate`` method to perform the computation and a ``reduce`` to reduce thread local histograms into a single histogram..
