=======================
Source Code Conventions
=======================

The guidelines below should be followed for any new code added to **freud**.

------

Naming Conventions
==================

The following conventions should apply to Python, Cython, and C++ code.

-  Variable names use :code:`lower_case_with_underscores`
-  Function and method names use :code:`lowerCaseWithNoUnderscores`
-  Class names use :code:`CapWords`

------

Indentation
===========

-  Spaces, not tabs, must be used for indentation
-  *4 spaces* are required per level of indentation and continuation lines

------

Python
======

Code in **freud** should follow
`PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_, as well as the
following guidelines. Anything listed here takes precedence over PEP 8,
but try to deviate as little as possible from PEP 8. When in doubt,
follow these guidelines over PEP 8.

During continuous integration (CI), all Python and Cython code in **freud** is
tested with `flake8 <http://flake8.pycqa.org/>`_ to ensure PEP 8 compliance.
It is strongly recommended to
`set up a pre-commit hook <http://flake8.pycqa.org/en/latest/user/using-hooks.html>`_
to ensure code is compliant before pushing to the repository:

.. code-block:: bash

    flake8 --install-hook git
    git config --bool flake8.strict true

Source
------

- All code should be contained in Cython files
- Python .py files are reserved for module level docstrings and minor
  miscellaneous tasks for, *e.g*, backwards compatibility.
- Semicolons should not be used to mark the end of lines in Python.


Documentation Comments
----------------------

-  Documentation is generated using `sphinx <http://www.sphinx-doc.org/en/stable/index.html>`_.
-  The documentation should be written according to the `Google Python Style Guide <https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings>`_.
-  A few specific notes:

   - The shapes of NumPy arrays should be documented as part of the type in the
     following manner::

        points ((:math:`N_{points}`, 3) :class:`numpy.ndarray`):

   - Constructors should be documented at the class level.
   - Class attributes (*including properties*) should be documented as class
     attributes within the class-level docstring.
   - Optional arguments should be documented as such within the type after the
     actual type, and the default value should be included within the
     description::

        box (:class:`freud.box.Box`, optional): Simulation box (Default value = None).

   - Properties that are settable should be documented the same way as optional
     arguments: :code:`Lx (float, settable): Length in x`.

-  All docstrings should be contained within the Cython files.
-  If you copy an existing file as a template, **make sure to modify the comments
   to reflect the new file**.
-  Docstrings should demonstrate how to use the code with an example. Liberal
   addition of examples is encouraged.

------

C++
===

C++ code should follow the result of running :code:`clang-format-6.0` with the style specified in the file :code:`.clang-format`.
Please refer to `Clang Format 6 <http://releases.llvm.org/6.0.1/tools/clang/docs/ClangFormatStyleOptions.html>`_ for details.

When in doubt, run :code:`clang-format -style=file FILE_WITH_YOUR_CODE` in the top directory of the **freud** repository.
If installing :code:`clang-format` is not a viable option, the :code:`check-style` step of
continuous integration (CI) contains the information on the correctness of the style.

Source
------

-  TBB sections should use lambdas, not functors (see
   `this tutorial <https://software.intel.com/en-us/blogs/2009/08/03/parallel_for-is-easier-with-lambdas-intel-threading-building-blocks>`_).

.. code-block:: c++

    void someFunction(float some_var, float other_var)
    {
        // code before parallel section
        parallel_for(blocked_range<size_t>(0, n), [=](const blocked_range<size_t>& r) {
            // do stuff
        });
    }

Documentation Comments
----------------------
-  Add explanatory comments throughout your code.
