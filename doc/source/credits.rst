Credits
=======

freud Developers
----------------

The following people contributed to the development of freud.

Eric Harper, University of Michigan - **Former lead developer**

* Added TBB parallelism.
* Wrote PMFT module.
* Added NearestNeighbors (since removed).
* Wrote RDF.
* Added bonding module (since removed).
* Added cubatic order parameter.
* Added hexatic order parameter.
* Added Pairing2D (since removed).
* Created common array conversion logic.

Joshua A. Anderson, University of Michigan - **Creator**

* Initial design and implementation.
* Wrote LinkCell and IteratorLinkCell.
* Wrote GaussianDensity, LocalDensity.
* Added parallel module.
* Added indexing modules (since removed).
* Wrote Cluster and ClusterProperties modules.

Matthew Spellings - **Former lead developer**

* Added generic neighbor list.
* Enabled neighbor list usage across freud modules.
* Added correlation functions.
* Added LocalDescriptors class.
* Added interface module.

Erin Teich

* Wrote environment matching (MatchEnv) class.
* Wrote BondOrder class (with Julia Dshemuchadse).
* Wrote AngularSeparation class (with Andrew Karas).
* Contributed to LocalQl development.
* Wrote LocalBondProjection class.

M. Eric Irrgang

* Authored kspace module (since removed).
* Fixed numerous bugs.
* Contributed to freud.shape (since removed).

Chrisy Du

* Authored Steinhardt order parameter classes.
* Fixed support for triclinic boxes.

Antonio Osorio

* Developed TrajectoryXML class.
* Various bug fixes.
* OpenMP support.

Vyas Ramasubramani - **Lead developer**

* Ensured PEP8 compliance.
* Added CircleCI continuous integration support.
* Create environment module and refactored order module.
* Rewrote most of freud docs, including order, density, and environment modules.
* Fixed nematic order parameter.
* Add properties for accessing class members.
* Various minor bug fixes.
* Refactored PMFT code.
* Refactored Steinhardt order parameter code.
* Wrote numerous examples of freud usage.
* Rewrote most of freud tests.
* Replaced CMake-based installation with setup.py using Cython.
* Add code coverage metrics.
* Added support for installing from PyPI, including ensuring that NumPy is installed.
* Converted all docstrings to Google format, fixed various incorrect docs.
* Debugged and added rotational autocorrelation code.
* Added MSD module.
* Wrote NeighborQuery, _QueryArgs, NeighborQueryResult classes.
* Wrote neighbor iterator infrastructure.
* Wrote PairCompute and SpatialHistogram parent classes.
* Wrote ManagedArray class.
* Wrote C++ histogram-related classes.
* Initial design of freud 2.0 API (NeighborQuery objects, neighbor computations, histograms).
* Standardized neighbor API in Python to use dictionaries of arguments or NeighborList objects for all pair computations.
* Standardized all attribute access into C++ with Python properties.
* Standardized variable naming of points/query\_points across all of freud.
* Standardized vector directionality in computes.
* Enabled usage of quaternions in place of angles for orientations in 2D PMFT calculations.
* Wrote new freud 2.0 compute APIs based on neighbor\_query objects and neighbors as either dictionaries or NeighborLists.
* Rewrote MatchEnv code to fit freud 2.0 API, splitting it into 3 separate calculations and rewriting internals using NeighborQuery objects.
* Wrote tutorial and reference sections of documentation.
* Unified util and common packages.
* Rewrote all docstrings in the package for freud 2.0.
* Changed Cubatic to use Mersenne Twisters for rng.

Bradley Dice - **Lead developer**

* Cleaned up various docstrings.
* Fixed bugs in HexOrderParameter.
* Cleaned up testing code.
* Added bumpversion support.
* Reduced all compile warnings.
* Added Python interface for box periodicity.
* Added Voronoi support for neighbor lists across periodic boundaries.
* Added Voronoi weights for 3D.
* Added Voronoi cell volume computation.
* Incorporated internal BiMap class for Boost removal.
* Wrote numerous examples of freud usage.
* Added some freud tests.
* Added ReadTheDocs support.
* Rewrote interface module into pure Cython.
* Added box duck-typing.
* Removed nose from unit testing.
* Use lambda function for parallelizing CorrelationFunction with TBB.
* Finalized boost removal.
* Wrote AABBQuery class.
* Consolidated cluster module functionality.
* Rewrote SolidLiquid order parameter class.
* Updated AngularSeparation class.
* Rewrote Voronoi implementation to leverage voro++.
* Implemented Voronoi bond weighting to enable Minkowski structure metrics.
* Refactored methods in Box and PeriodicBuffer for v2.0.
* Added checks to C++ for 2D boxes where required.
* Refactored cluster module.
* Standardized vector directionality in computes.
* NeighborQuery support to ClusterProperties, GaussianDensity, Voronoi, PeriodicBuffer, Interface.
* Standardized APIs for order parameters.
* Added radius of gyration to ClusterProperties.
* Improved Voronoi plotting code.
* Corrected number of points/query points in LocalDensity.
* Made PeriodicBuffer inherit from _Compute.

Richmond Newman

* Developed the freud box.
* Solid liquid order parameter.

Carl Simon Adorf

* Developed the Python box module.

Jens Glaser

* Wrote kspace front-end (since removed).
* Modified kspace module (since removed).
* Wrote Nematic order parameter class.

Benjamin Schultz

* Wrote Voronoi class.
* Fix normalization in GaussianDensity.
* Bug fixes in shape module (since removed).

Bryan VanSaders

* Make Cython catch C++ exceptions.
* Add shiftvec option to PMFT.

Ryan Marson

* Various GaussianDensity bugfixes.

Yina Geng

* Co-wrote Voronoi neighbor list module.
* Add properties for accessing class members.

Carolyn Phillips

* Initial design and implementation.
* Package name.

Ben Swerdlow

* Documentation and installation improvements.

James Antonaglia

* Added number of neighbors as an argument to HexOrderParameter.
* Bugfixes.
* Analysis of deprecated kspace module.

Mayank Agrawal

* Co-wrote Voronoi neighbor list module.

William Zygmunt

* Helped with Boost removal.

Greg van Anders

* Bugfixes for CMake and SSE2 installation instructions.

James Proctor

* Cythonization of the cluster module.

Rose Cersonsky

* Enabled TBB-parallelism in density module.
* Fixed how C++ arrays were pulled into Cython.

Wenbo Shen

* Translational order parameter.

Andrew Karas

* Angular separation.
* Wrote reference implementation for rotational autocorrelation.

Paul Dodd

* Fixed CorrelationFunction namespace, added ComputeOCF class for TBB parallelization.

Tim Moore

* Added optional rmin argument to density.RDF.
* Enabled NeighborList indexing.

Alex Dutton

* BiMap class for MatchEnv.

Matthew Palathingal

* Replaced use of boost shared arrays with shared ptr in Cython.
* Helped incorporate BiMap class into MatchEnv.

Kelly Wang

* Enabled NeighborList indexing.

Yezhi Jin

* Added support for 2D arrays in the Python interface to Box functions.
* Rewrote Voronoi implementation to leverage voro++.
* Implemented Voronoi bond weighting to enable Minkowski structure metrics.

Brandon Butler

* Rewrote Steinhardt order parameter.

Jin Soo Ihm

* Added benchmarks.
* Contributed to NeighborQuery classes.
* Refactored C++ to perform neighbor queries on-the-fly.
* Added plotting functions to analysis classes.
* Wrote RawPoints class.
* Created Compute parent class with decorators to ensure properties have been computed.
* Updated common array conversion logic.
* Added many validation tests.

Source code
-----------

Eigen (http://eigen.tuxfamily.org/) is included as a git submodule in freud.
Eigen is made available under the Mozilla Public License v2.0
(http://mozilla.org/MPL/2.0/). Its linear algebra routines are used for
various tasks including the computation of eigenvalues and eigenvectors.

fsph (https://github.com/glotzerlab/fsph) is included as a git submodule in
freud. It is used for the calculation of spherical harmonics. fsph is made
available under the MIT license::

    Copyright (c) 2016 The Regents of the University of Michigan

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
