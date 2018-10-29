# Change Log
The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## v0.11.3 - 2018-10-18

### Fixed
* Linux wheels are now pushed to the real PyPI server instead of the test server.
* macOS deployment pyenv requires patch versions to be specified.

## v0.11.2 - 2018-10-18

### Fixed
* Error in Python versions in macOS automatic deployment.

## v0.11.1 - 2018-10-18

### Added
* PyPI builds automatically deploy for Mac and Linux.

### Changed
* macOS deployment target is now 10.12 instead of 10.9 to ensure TBB compatibility.
* Unwrapping positions with images is now vectorized.
* Minor documentation fixes.

### Fixed
* TBB includes were not always detected correctly by setup.py.

## v0.11.0 - 2018-09-27

### Added
* Example notebooks are now shown in the documentation.
* Many unit tests were added.
* New class: `freud.environment.LocalBondProjection`.
* `freud` is now available on the Python Package Index (PyPI) as `freud-analysis`.

### Changed
* Documentation was revised for several modules.
* New class `freud.box.ParticleBuffer` was adapted from the previous `VoronoiBuffer` to include support for triclinic boxes.
* The `bond` and `pmft` modules verify system dimensionality matches the coordinate system used.
* Minor optimization: arrays are reduced across threads only when necessary.

### Fixed
* NumPy arrays of lengths 2, 3, 6 are now correctly ducktyped into boxes.
* Removed internal use of deprecated code.
* C++ code using `uint` has been changed to `unsigned int`, to improve compiler compatibility.

### Deprecated
* In `freud.locality.LinkCell`, `computeCellList()` has been replaced by `compute()`.

### Removed
* The `kspace` module has been removed.

## v0.10.0 - 2018-08-27

### Added
* Codecov to track test coverage.
* Properties were added to MatchEnv, AngularSeparation, Cubatic/Nematic order parameters, Voronoi.

### Changed
* freud uses Cython and setup.py instead of CMake for installation.
* Properties (not get functions) are the official way to access computed results.
* Interface module has been improved significantly.
* density.FloatCF, density.ComplexCF, order parameter documentation is improved.
* Many compute methods now use points, orientations from ref\_points, ref\_orientations if not provided.
* Reset methods have been renamed to `reset`.

### Fixed
* `kspace` module had a missing factor of pi in the volume calculation of `FTsphere`.

### Deprecated
* Get functions have been deprecated.
* Setter methods have been deprecated.
* Reduce methods are called internally, so the user-facing methods have been deprecated.

### Removed
* GaussianDensity.resetDensity() is called internally.

## v0.9.0 - 2018-07-30

### Added
* Allow specification of rmin for LocalWl (previously was only possible for LocalQl).
* New environment module. Contains classes split from the order module.
* Box duck-typing: methods accepting a box argument will convert box-like objects into freud.box.Box objects.
* All Python/Cython code is now validated with flake8 during continuous integration.

### Changed
* Refactoring of LocalQl and LocalWl Steinhardt order parameters.
* MatchEnv uses BiMap instead of boost::bimap.
* All boost shared\_arrays have been replaced with std::shared\_ptr.
* Replaced boost geometry with standard containers in brute force registration code.
* NearestNeighbors automatically uses ref\_points as the points if points are not provided.
* Box::unwrap and Box::wrap return the vectors after updating.
* Everything other than true order parameters moved from Order module to Environment module.
* Use lambda function in parallel\_for in CorrelationFunction.
* Tests no longer depend on nose. Python's unittest is used instead.
* Vastly improved documentation clarity and correctness across all modules.
* Docstrings are now in Google format. The developer guide offers guidance for module authors.

### Fixed
* Fixed LocalDescriptors producing NaN's in some cases.
* Fixed cython passing C++ the default argument force\_resize to NeighborList::resize.
* Standardize freud.common.convert\_array error message.

### Removed
* Boost is no longer needed to build or run freud.
* Removed undocumented shapesplit module.
* Removed extra argument from TransOrderParam in C++.

## v0.8.2 - 2018-06-07

### Added
* Allow specification of maximum number of neighbors to use when computing LocalDescriptors

### Changed
* Using the default neighbor list with LocalDescriptors requires specifying the
  precompute argument
* Updated and improved tests
* Cleaned AngularSeparation module and documentation

## v0.8.1 - 2018-05-09

### Fixed
* Memory issue in nlist resolved

## v0.8.0 - 2018-04-06

### Added
* Voronoi neighborlist now includes periodic neighbors
* Voronoi neighborlist computes weight according to the facet area in 3D
* Box module exposes `getImage(vec)`
* Voronoi module can compute and return cell volumes/areas

### Changed
* Cluster module supports box argument in compute methods.
* Refactored C++ code to reduce extraneous #includes
* Refactored PMFT code
* Refactored box module to remove unused methods
* Resolved bug in `kspace.AnalyzeSFactor3D`

### Deprecated
* Box module `getCoordinates()` in favor of duplicate `box.makeCoordinates()`

### Removed
* Removed deprecated API for ComplexWRDF and FloatWRDF

## v0.7.0 - 2018-03-02

### Added
* Added nematic order parameter
* Added optional rmin argument to density.RDF
* Added credits file
* Wrote development guide
* Added Python interface for box periodicity

### Changed
* Various bug fixes and code cleaning
* Fixed all compile-time warnings
* Ensured PEP 8 compliance everywhere
* Minimized boost dependence
* Many documentation rewrites
* Wrote development guide
* Made tests deterministic (seeded RNGs)
* Removed deprecated Box API warnings
* Standardized numpy usage

## v0.6.4 - 2018-02-05

* Added a generic neighbor list interface
* Set up CircleCI for continuous integration
* Set up documentation on ReadTheDocs
* Added bumpversion support
* Various bug fixes
* Added python-style properties for accessing data
* Fixed issues with voronoi neighbor list

## v0.6.0

* trajectory module removed
* box constructor API updated
* PMFTXYZ API updated to take in more quaternions for `face_orientations`, or have a sensible default value
* NearestNeighbors:
    - over-expanding box fixed
    - strict rmax mode added
    - ability to get wrapped vectors added
    - minor updates to C-API to return full lists from C
* Addition of Bonding modules
* Addition of local environment matching

## v0.5.0

* Replace boost::shared\_array with std::shared\_ptr (C++ 11)
* Moved all tbb template classes to lambda expressions
* Moved trajectory.Box to box.Box
* trajectory is deprecated
* Fixed Bond Order Diagram and allow for global, local, or orientation correlation
* Added python-level voronoi calculation
* Fixed issues with compiling on OS X, including against conda python installs
* Added code to compute bonds between particles in various coordinate systems

## v0.4.1
* PMFT: Fixed issue involving binning of angles correctly
* PMFT: Fixed issue in R12 which prevented compute/accumulate from being called with non-flattened arrays
* PMFT: Updated xyz api to allow simpler symmetric orientations to be supplied
* PMFT: Updated pmftXY2D api
* PMFT: Histograms are properly normalized, allowing for comparison between systems without needing to "zero" the system
* fsph: Added library to calculate spherical harmonics via cython
* Local Descriptors: Uses fsph, updates to API
* Parallel: Added default behavior to setNumThreads and added context manager


## v0.4.0

* Add compiler flags for C++11 features
* Added Saru RNG (specifically for Cubatic Order Parameter, available to all)
* Cubatic Order Parameter
* Rank 4 tensor struct
* Environment Matching/Cluster Environment
* Shape aware fourier transform for structure factor calculation
* Added deprecation warnings; use python -W once to check warnings
* Added Change Log
* Moved all documentation in Sphinx; documentation improvements
* C++ wrapping moved from Boost to Cython
* Itercell only available in python 3.x
* PMFT:
    - XY
    - XYZ with symmetry, API updated
    - R, T1, T2
    - X, Y, T2
* viz removed (is not compatible with cython)

## No change logs prior to v0.4.0 ##
