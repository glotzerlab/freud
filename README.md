freud {#mainpage}
===============================================

# Requirements {#requirements_section}

Numpy is **required** to build freud.

# Documentation {#documentation_section}

To view the full documentation run the following commands in the source directory:

~~~
# Linux
doxygen
firefox doc/html/index.html

# Mac
/Applications/Doxygen.app/Contents/Resources/doxygen
open doc/html/index.html
~~~

# Build {#build_section}

Make a build directory, preferably one external to the source dir.

~~~
cd build
cmake ~/source
make install -j6
~~~

Make sure to set the install path to something like `~/local`

# Tests {#tests_section}

Run all unit tests with nosetests in the source directory. To add a test, simply add a file to the `tests` directory,
and nosetests will automatically discover it. See http://pythontesting.net/framework/nose/nose-introduction/ for
an introduction to writing nose tests.

~~~
cd source
nosetests
~~~

# Available Modules #

The following modules are available to use:

## Bootstrap ##

Perform a statistical bootstrap on an array of data. Currently requires the data to be unsigned ints.

Maintainer: Harper

## Cluster ##

Maintainer: ?

## Density ##

Perform analysis involving density of a system:

1. Gaussian Density: Perform a gaussian blur
2. Local Density: Calculate the local density out to an \\(r_{cut}\\) value
3. RDF: Calculate the radial distribution function
4. Correlation Function: Calculate a generic correlation function using real or complex values

Maintainers: Harper, Matthew

## Interface ##

Maintainer: ?

## KSpace ##

Maintainer: Irrgang

## Lindemann ##

Calculate the lindemann quotient for a series of frames.

*Note: Hasn't been updated in a while. Could be improved. May need refactored and require more validation*

Maintainer: Harper

## Locality ##

Perform analysis involving a particles immediate neighborhood

1. LinkCell: Module for creating a cell list
2. Iterator Link Cell: Returns an iterator to iterate over a cell
3. Nearest Neighbors: Return \\(k\\) nearest neighbors for a particle \\(i\\)

Maintainers: Harper, Josh

## Order ##

Calculate various order parameters

1. Hexatic \\( \left( \psi \right) \\) Order Parameter
2. Local Descriptors: ?

Maintainers: Harper, Josh, Matthew

## Pairing ##

Determine if two particles in 2D have paired.

Maintainer: Harper

## Parallel ##

Methods to allow parallel computation via TBB

Maintainer: Josh

## PMFT ##

Perform PMF(T) analysis on a system

1. pmfXYZ: Integrated 3D PMF analysis
2. pmfXY2D: Integrated 2D PMF analysis
3. pmftXYT2D: Full 2D pmft binned for \\(\theta\\)
4. pmftXYTP2D: Full 2D pmft binned for \\( \theta = \phi_1 + \phi_2 \\)
5. pmftXYTM2D: Full 2D pmft binned for \\( \theta = \phi_1 - \phi_2 \\)
6. pmftRPM: Full 2D pmft binned for \\( \theta_P = \phi_1 + \phi_2; \theta_M = \phi_1 - \phi_2 \\)

Maintainer: Harper

## QT ##

Helpers for the freud viz project. Will be moved to separate project

Maintainer: Josh

## Spherical Harmonic Order Parameter ##

?

Maintainer: Richmond, Chrissy

## Split ##

Method to split a shape into \\(n\\) subshapes. Useful to combine with pmfXYZ to perform further integration, combine with order parameter to determine the order of e.g. rectangles if \\(\psi_4\\) is desired.

Maintainer: Harper

## Trajectory ##

All modules to load in "Glotzer Group" trajectories:

1. TrajectoryVMD
2. TrajectoryXML
3. TrajectoryXMLDCD
4. TrajectoryPOS *Note* minimal support
5. TrajectoryHOOMD
6. TrajectoryDISCMC

Also contains the box class (in c++)

Maintainer: Harper, Josh

## Voronoi ##

Calculate Voronoi cells and other stuff.

Maintainer: Ben

## Wigner3j ##

?

Maintainer: ?