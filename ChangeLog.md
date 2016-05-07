# Change Log

## v0.5.0

Replace boost::shared_array with std::shared_ptr (C++ 11)

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
