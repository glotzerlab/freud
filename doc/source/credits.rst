Credits
=======

freud Developers
----------------

The following people contributed to the development of freud.

Eric Harper, University of Michigan - **Former lead developer**

* TBB parallelism.
* PMFT module.
* NearestNeighbors.
* RDF.
* Bonding module (since removed).
* Cubatic order parameter.
* Hexatic order parameter.
* Pairing2D (since removed).

Joshua A. Anderson, University of Michigan - **Creator**

* Initial design and implementation.
* IteratorLinkCell.
* LinkCell.
* Various density modules.
* freud.parallel.
* Indexing modules.
* cluster.pxi.

Matthew Spellings - **Former lead developer**

* Added generic neighbor list.
* Enabled neighbor list usage across freud modules.
* Correlation functions.
* LocalDescriptors class.
* interface.pxi.

Erin Teich

* Wrote environment matching module.
* BondOrder (with Julia Dshemuchadse).
* Angular separation (with Andrew Karas).
* Contributed to LocalQl development.
* Wrote LocalBondProjection module.

M. Eric Irrgang

* Authored (now removed) kspace code.
* Numerous bug fixes.
* Various contributions to freud.shape.

Chrisy Du

* Authored all Steinhardt order parameters.
* Fixed support for triclinic boxes.

Antonio Osorio

* Developed TrajectoryXML class.
* Various bug fixes.
* OpenMP support.

Vyas Ramasubramani - **Lead developer**

* Ensured pep8 compliance.
* Added CircleCI continuous integration support.
* Rewrote docs.
* Fixed nematic order parameter.
* Add properties for accessing class members.
* Various minor bug fixes.
* Refactored PMFT code.
* Refactored Steinhardt order parameter code.
* Wrote numerous examples of freud usage.
* Rewrote most of freud tests.
* Replaced CMake-based installation with setup.py using Cython.
* Split non-order parameters out of order module into separate environment module..
* Rewrote documentation for order, density, and environment modules.
* Add code coverage metrics.
* Added support for PyPI, including ensuring that NumPy is installed.
* Converted all docstrings to Google format, fixed various incorrect docs.

Bradley Dice - **Lead developer**

* Cleaned up various docstrings.
* HexOrderParameter bug fixes.
* Cleaned up testing code.
* Bumpversion support.
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
* Proper box duck-typing.
* Removed nose from unit testing.
* Use lambda function for parallelizing CorrelationFunction with TBB.
* Finalized boost removal.

Richmond Newman

* Developed the freud box.
* Solid liquid order parameter.

Carl Simon Adorf

* Developed the python box module.

Jens Glaser

* Wrote kspace.pxi front-end.
* Modifications to kspace module.
* Nematic order parameter.

Benjamin Schultz

* Wrote Voronoi module.
* Fix normalization in GaussianDensity.
* Bugfixes in freud.shape.

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

Source code
-----------

Eigen (http://eigen.tuxfamily.org/) is included as a git submodule in freud.
Eigen is made available under the Mozilla Public License v.2.0
(http://mozilla.org/MPL/2.0/). Its linear algebra routines are used for
various tasks including the computation of eigenvalues and eigenvectors.

fsph (https://bitbucket.org/glotzer/fsph) is included as a git submodule in
freud. fsph is made available under the MIT license. It is used for the
calculation of spherical harmonics, which are then used in the calculation of
various order parameters, under the following license::

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
