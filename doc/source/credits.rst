Credits
=======

freud Developers
----------------

The following people contributed to the development of freud.

Eric Harper, University of Michigan - **Former lead developer**
 * PMFT module
 * NearestNeighbors
 * RDF
 * Bonding module
 * Cubatic OP
 * Hexatic OP
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
 * Environment matching
 * BondOrder
 * Angular separation

M. Eric Irrgang
 * Authored kspace CPP code

Chrisy Du
 * Authored all Steinhardt order parameters

Antonio Osorio

Vyas Ramasubramani - **Lead developer**
 * Ensured pep8 compliance
 * Added CircleCI continuous integration support
 * Rewrote docs
 * Fixed nematic OP
 * Add properties for accessing class members
 * Various minor bug fixes

Bradley Dice - **Lead developer**
 * Cleaned up various docstrings
 * HexOrderParameter bug fixes
 * Cleaned up testing code
 * Bumpversion support
 * Reduced all compile warnings

Richmond Newman
 * Developed the freud box
 * Solid liquid order parameter

Carl Simon Adorf
 * Developed the python box module

Jens Glaser
 * Wrote kspace.pxi front-end
 * Nematic OP

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

Source code
-----------

Eigen (http://eigen.tuxfamily.org/) is embedded in freud's package and is
made available under the Mozilla Public License v.2.0
(http://mozilla.org/MPL/2.0/). It's linear algebra routines are used for
various tasks including the computation of eigenvalues and eigenvectors.

fsph (https://bitbucket.org/glotzer/fsph) is embedded in freud's package
and is made available under the MIT license. It is used for the calculation
of spherical harmonics, which are then used in the calculation of various
order parameters.
