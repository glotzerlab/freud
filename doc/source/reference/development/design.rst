=================
Design Principles
=================

Vision
======

The **freud** library is designed to be a powerful and flexible library
for the analysis of simulation output. To support a variety of
analysis routines, **freud** places few restrictions on its components.
The primary requirement for an analysis routine in **freud** is that it
should be substantially computationally intensive so as to require
coding up in C++: **all freud code should be composed of fast C++
routines operating on systems of particles in periodic boxes.** To
remain easy-to-use, all C++ modules should be wrapped in Python
code so they can be easily accessed from Python scripts or through
a Python interpreter.

In order to achieve this goal, **freud** takes the following viewpoints:

* **freud** works directly with `NumPy <https://numpy.org/>`__
  arrays to retain maximum flexibility. Integrations with other tools
  should be performed via the common data representations of NumPy arrays.
* For ease of maintenance, **freud** uses Git for version control;
  GitHub for code hosting and issue tracking; and the PEP 8
  standard for code, stressing explicitly written code which is easy
  to read.
* To ensure correctness, **freud** employs unit testing using the
  Python :mod:`pytest` framework. In addition, **freud** utilizes
  `CircleCI <https://circleci.com>`__ for continuous integration to
  ensure that all of its code works correctly and that any changes or
  new features do not break existing functionality.

Language choices
================

The **freud** library is written in two languages: Python and C++. C++ allows for
powerful, fast code execution while Python allows for easy, flexible
use. Intel Threading Building Blocks parallelism provides further power to
C++ code. The C++ code is wrapped with Cython, allowing for user
interaction in Python. NumPy provides the basic data structures in
**freud**, which are commonly used in other Python plotting libraries and
packages.

Unit Tests
==========

All modules should include a set of unit tests which test the correct
behavior of the module. These tests should be simple and short, testing
a single function each, and completing as quickly as possible
(ideally < 10 sec, but times up to a minute are acceptable if justified).

Benchmarks
==========

Modules can be benchmarked in the following way.
The following code is an example benchmark for the :code:`freud.density.RDF` module.

.. literalinclude:: ../../../../benchmarks/benchmark_density_RDF.py
   :language: python
   :linenos:

in a file :code:`benchmark_density_RDF.py` in the :code:`benchmarks` directory.
More examples can be found in the :code:`benchmarks` directory.
The runtime of :code:`BenchmarkDensityRDF.bench_run` will be timed for :code:`number`
of times on the input sizes of :code:`Ns`. Its runtime with respect to the number of threads
will also be measured. Benchmarks are run as a part of continuous integration,
with performance comparisons between the current commit and the master branch.

Make Execution Explicit
=======================

While it is tempting to make your code do things "automatically", such
as have a calculate method find all :code:`_calc` methods in a class, call
them, and add their returns to a dictionary to return to the user, it is
preferred in **freud** to execute code explicitly. This helps avoid issues
with debugging and undocumented behavior:

.. code-block:: python

    # this is bad
    class SomeFreudClass(object):
        def __init__(self, **kwargs):
            for key in kwargs.keys:
                setattr(self, key, kwargs[key])

    # this is good
    class SomeOtherFreudClass(object):
        def __init__(self, x=None, y=None):
            self.x = x
            self.y = y

Code Duplication
================

When possible, code should not be duplicated. However, being explicit is
more important. In **freud** this translates to many of the inner loops of
functions being very similar:

.. code-block:: c++

    // somewhere deep in function_a
    for (int i = 0; i < n; i++)
        {
        vec3[float] pos_i = position[i];
        for (int j = 0; j < n; j++)
            {
            pos_j = = position[j];
            // more calls here
            }
        }

    // somewhere deep in function_b
    for (int i = 0; i < n; i++)
        {
        vec3[float] pos_i = position[i];
        for (int j = 0; j < n; j++)
            {
            pos_j = = position[j];
            // more calls here
            }
        }

While it *might* be possible to figure out a way to create a base C++
class all such classes inherit from, run through positions, call a
calculation, and return, this would be rather complicated. Additionally,
any changes to the internals of the code may result in performance
penalties, difficulty in debugging, etc. As before, being explicit is
better.

However, if you have a class which has a number of methods, each of
which requires the calling of a function, this function should be
written as its own method (instead of being copy-pasted into each
method) as is typical in object-oriented programming.

Python vs. Cython vs. C++
=========================

The **freud** library is meant to leverage the power of C++ code imbued with
parallel processing power from TBB with the ease of writing Python code.
The bulk of your calculations should take place in C++, as shown in the
snippet below:

.. code-block:: python

    # this is bad
    def badHeavyLiftingInPython(positions):
        # check that positions are fine
        for i, pos_i in enumerate(positions):
            for j, pos_j in enumerate(positions):
                if i != j:
                    r_ij = pos_j - pos_i
                    # ...
                    computed_array[i] += some_val
        return computed_array

    # this is good
    def goodHeavyLiftingInCPlusPlus(positions):
        # check that positions are fine
        cplusplus_heavy_function(computed_array, positions, len(pos))
        return computed_array

In the C++ code, implement the heavy lifting function called above from Python:

.. code-block:: c++

    void cplusplus_heavy_function(float* computed_array,
                                  float* positions,
                                  int n)
        {
        for (int i = 0; i < n; i++)
            {
            for (int j = 0; j < n; j++)
                {
                if (i != j)
                    {
                    r_ij = pos_j - pos_i;
                    // ...
                    computed_array[i] += some_val;
                    }
                }
            }
        }

Some functions may be necessary to write at the Python level due to a Python
library not having an equivalent C++ library, complexity of coding, etc. In
this case, the code should be written in Cython and a *reasonable* attempt
to optimize the code should be made.
