============
Introduction
============

The **freud** library is a Python package for analyzing particle simulations.
The package is designed to directly use numerical arrays of data, making it easy to use for a wide range of use-cases.
The most common use-case of **freud** is for computing quantities from molecular dynamics simulation trajectories, but it can be used for analyzing essentially any type of simulation information.
By operating directly on numerical array data, **freud** allows users to parse custom simulation outputs into a suitable structure for input, rather than relying specific file types or data structures.

The core of **freud** is analysis of periodic systems, which are represented through the freud ``Box`` class.
The ``Box`` supports arbitrary triclinic systems for maximum flexibility, and is used throughout the package to ensure consistent treatment of these systems.
The package's many methods are encapsulated in various *compute classes*, which perform computations and populate class attributes for access.
Of particular note are the various computations based on nearest neighbor finding in order to characterize particle environments.
Such methods are simplified and accelerated through a centralized neighbor finding interface defined in the ``NeighborQuery`` family of classes in the ``locality`` module of freud.
