Credits
=======

freud Developers
----------------

The following people contributed to the development of freud.

Eric Harper, University of Michigan - **Former lead developer**

* TBB parallelism
* PMFT module
* NearestNeighbors
* RDF
* Bonding module
* Cubatic order parameter
* Hexatic order parameter
* Pairing2D

Joshua A. Anderson, University of Michigan - **Creator**

* Initial design and implementation
* IteratorLinkCell
* LinkCell
* Various density modules
* freud.parallel
* Indexing modules
* cluster.pxi

Matthew Spellings - **Former lead developer**

* Added generic neighbor list
* Enabled neighbor list usage across freud modules
* Correlation functions
* LocalDescriptors class
* interface.pxi

Erin Teich

* Wrote environment matching module
* BondOrder (with Julia Dshemuchadse)
* Angular separation (with Andrew Karas)
* Contributed to LocalQl development

M. Eric Irrgang

* Authored kspace CPP code

Chrisy Du

* Authored all Steinhardt order parameters

Antonio Osorio

Vyas Ramasubramani - **Lead developer**

* Ensured pep8 compliance
* Added CircleCI continuous integration support
* Rewrote docs
* Fixed nematic order parameter
* Add properties for accessing class members
* Various minor bug fixes
* Refactored PMFT code
* Refactored Steinhardt order parameter code

Bradley Dice - **Lead developer**

* Cleaned up various docstrings
* HexOrderParameter bug fixes
* Cleaned up testing code
* Bumpversion support
* Reduced all compile warnings
* Added Python interface for box periodicity
* Added Voronoi support for neighbor lists across periodic boundaries
* Added Voronoi weights for 3D
* Added Voronoi cell volume computation

Richmond Newman

* Developed the freud box
* Solid liquid order parameter

Carl Simon Adorf

* Developed the python box module

Jens Glaser

* Wrote kspace.pxi front-end
* Nematic order parameter

Benjamin Schultz

* Wrote Voronoi module

Bryan VanSaders

Ryan Marson

Tom Grubb

Yina Geng

* Co-wrote Voronoi neighbor list module
* Add properties for accessing class members

Carolyn Phillips

* Initial design and implementation
* Package name

Ben Swerdlow

James Antonaglia

Mayank Agrawal

* Co-wrote Voronoi neighbor list module

William Zygmunt

Greg van Anders

James Proctor

Rose Cersonsky

Wenbo Shen

Andrew Karas

* Angular separation

Paul Dodd

Tim Moore

* Added optional rmin argument to density.RDF

Michael Engel

* Translational order parameter

Alex Dutton

* BiMap class for MatchEnv

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
