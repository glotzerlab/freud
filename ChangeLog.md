# Change Log

## v0.5.0

* Replace boost::shared_array with std::shared_ptr (C++ 11)
* Moved all tbb template classes to lambda expressions
* Moved trajectory.Box to box.Box
* trajectory is deprecated
* Fixed Bond Order Diagram and allow for global, local, or orientation correlation
* Added python-level voronoi calculation
* Fixed issues with compiling on OS X, including against conda python installs

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
