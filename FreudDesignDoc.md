# Design #

## Vision ##

The freud library is designed to be:

1. Powerful
2. Flexible
3. Maintainable

### Powerful ###

The amount of data produced by simulations is always increasing. By being powerful, freud allows users to analyze their
simulation data as fast as possible so that it can be used in real-time visualization and on-line simulation analysis.

### Flexible ###

The number of simulation packages, analysis packages, and other software packages keeps growing. Rather than attempt to
understand and interact with all of these packages, freud achieves flexibility by providing a simple Python interface
and making no assumptions regarding data, operating on and returning NumPy arrays to the user.

### Maintainable ###

Code which cannot be maintained is destined for obscurity. In order to be maintainable, freud uses Git for version
control; BitBucket for code hosting, issue tracking; the PEP8 standard for code, stressing explicitly written code which
is easy to read.

## Language choices ##

The freud library is written in two languages: Python and C++. C++ allows for powerful, fast code execution while
Python allows for easy, flexible use. Intel Thread Building Blocks parallelism provides further power to C++ code. The
C++ code is wrapped with Cython, allowing for user interaction in Python. NumPy provides the basic data structures in
freud, which are commonly used in other Python plotting libraries and packages.

## Unit Tests ##

All modules should include a set of unit tests which test the correct behavior of the module. These tests should be
simple and short, testing a single function each, and completing as quickly as possible (ideally < 10 sec, but times
up to a minute are acceptable if justified, documented, and by default skipped, functionality current TBD).
