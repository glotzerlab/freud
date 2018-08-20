.. _installation:

============
Installation
============

Installing freud
================

You can either install freud via `conda <http://conda.pydata.org/docs/>`_ or compile it from source.

Install via conda
-----------------

The code below will install freud from conda-forge.

.. code-block:: bash

    conda install -c conda-forge freud

Compile from source
-------------------

The following are **required** for installing freud:

- `Python <https://www.python.org/>`_ (2.7, 3.5, 3.6)
- `NumPy <http://www.numpy.org/>`_
- `Intel Threading Building Blocks <https://www.threadingbuildingblocks.org/>`_ (TBB)

The following are **optional** for installing freud:

- `Cython <http://cython.org/>`_: The freud repository contains Cython-generated :code:`*.cpp` files that can be used directly. However, Cython is necessary if you wish to recompile these files.

The code that follows creates a build directory inside the freud source directory and builds freud:

.. code-block:: bash

    git clone --recurse-submodules https://bitbucket.org/glotzer/freud.git
    cd freud
    python setup.py install

By default, freud installs to the `USER_SITE <https://docs.python.org/3/install/index.html>`_ directory, which is in ``~/.local`` on Linux and in ``~/Library`` on macOS.
:code:`USER_SITE` is on the Python search path by default, so there is no need to modify :code:`PYTHONPATH`.

.. note::

    freud makes use of submodules. If you ever wish to manually update these, you can execute:

    .. code-block:: bash

        git submodule update --init

Unit Tests
==========

The unit tests for freud are included in the repository and are configured to be run using the Python :py:mod:`unittest` library:

.. code-block:: bash

    # Run tests from the tests directory
    cd tests
    python -m unittest discover .

Note that because freud is designed to require installation to run (*i.e.* it cannot be run directly out of the build directory), importing freud from the root of the repository will fail because it will try and import the package folder.
As a result, unit tests must be run from outside the root directory if you wish to test the installed version of freud.
If you want to run tests within the root directory, you can instead build freud in place:

.. code-block:: bash

    # Run tests from the tests directory
    python setup.py build_ext --inplace

This build will place the necessary files alongside the freud source files so that freud can be imported from the root of the repository.

Documentation
=============

The documentation for freud is hosted online at `ReadTheDocs <https://freud.readthedocs.io/>`_, but you may also build the documentation yourself:

Building the documentation
--------------------------

The following are **required** for building freud documentation:

- `Sphinx <http://www.sphinx-doc.org/>`_

You can install sphinx using conda

.. code-block:: bash

    conda install sphinx

or from PyPi

.. code-block:: bash

    pip install sphinx

To build the documentation, run the following commands in the source directory:

.. code-block:: bash

    cd doc
    make html
    # Then open build/html/index.html

To build a PDF of the documentation (requires LaTeX and/or PDFLaTeX):

.. code-block:: bash

    cd doc
    make latexpdf
    # Then open build/latex/freud.pdf
