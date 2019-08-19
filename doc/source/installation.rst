.. _installation:

============
Installation
============

Installing freud
================

The freud library can be installed via `conda <https://conda.io/projects/conda/>`_ or pip, or compiled from source.

Install via conda
-----------------

The code below will install freud from `conda-forge <https://anaconda.org/conda-forge/freud>`_.

.. code-block:: bash

    conda install -c conda-forge freud

Install via pip
-----------------

The code below will install freud from `PyPI <https://pypi.org/project/freud-analysis/>`_.

.. code-block:: bash

    pip install freud-analysis

Compile from source
-------------------

The following are **required** for installing freud:

- `Python <https://www.python.org/>`_ (3.5+ required)
- `NumPy <http://www.numpy.org/>`_
- `Intel Threading Building Blocks <https://www.threadingbuildingblocks.org/>`_ (TBB)

The following are **optional** for installing freud:

- `Cython <http://cython.org/>`_ (0.28+ required): The freud repository contains Cython-generated :code:`*.cpp` files in the :code:`freud/` directory that can be used directly. However, Cython is necessary if you wish to recompile these files.

For conda users, these requirements can be met by installing the following packages from the `conda-forge channel <https://conda-forge.org/>`_:

.. code-block:: bash

    conda install -c conda-forge tbb tbb-devel numpy cython

The code that follows builds freud and installs it for all users (append `--user` if you wish to install it to your user site directory):

.. code-block:: bash

    git clone --recurse-submodules https://github.com/glotzerlab/freud.git
    cd freud
    python setup.py install

You can also build freud in place so that you can run from within the folder:

.. code-block:: bash

    # Run tests from the tests directory
    python setup.py build_ext --inplace

Building freud in place has certain advantages, since it does not affect your Python behavior except within the freud directory itself (where freud can be imported after building).
Additionally, due to limitations inherent to the distutils/setuptools infrastructure, building extension modules can only be parallelized using the build_ext subcommand of setup.py, not with install.
As a result, it will be faster to manually run build_ext and then install (which normally calls build_ext under the hood anyway) the built packages.
In general, the following options are available for setup.py in addition to the standard setuptools options (notes are included to indicate which options are only available for specific subcommands such as build_ext):

.. glossary::

    -\\-PRINT-WARNINGS
      Specify whether or not to print compilation warnings resulting from the build even if the build succeeds with no errors.

    -\\-ENABLE-CYTHON
      Rebuild the Cython-generated C++ files.
      If there are any unexpected issues with compiling the C++ shipped with the build, using this flag may help.
      It is also necessary any time modifications are made to the Cython files.

    -j
      Compile in parallel.
      This affects both the generation of C++ files from Cython files and the subsequent compilation of the source files.
      In the latter case, this option controls the number of Python modules that will be compiled in parallel.

    -\\-TBB-ROOT
      The root directory where TBB is installed.
      Useful if TBB is installed in a non-standard location or cannot be located by Python for some other reason.
      Note that this information can also be provided using the environment variable TBB_ROOT.
      The options --TBB-INCLUDE and --TBB-LINK will take precedence over --TBB-ROOT if both are specified.

    -\\-TBB-INCLUDE
      The directory where the TBB headers (*e.g.* tbb.h) are located.
      Useful if TBB is installed in a non-standard location or cannot be located by Python for some other reason.
      Note that this information can also be provided using the environment variable TBB_ROOT.
      The options --TBB-INCLUDE and --TBB-LINK will take precedence over --TBB-ROOT if both are specified.

    -\\-TBB-LINK
      The directory where the TBB shared library (*e.g.* libtbb.so or libtbb.dylib) is located.
      Useful if TBB is installed in a non-standard location or cannot be located by Python for some other reason.
      Note that this information can also be provided using the environment variable TBB_ROOT.
      The options --TBB-INCLUDE and --TBB-LINK will take precedence over --TBB-ROOT if both are specified.

The following additional arguments are primarily useful for developers:

.. glossary::

    -\\-COVERAGE
      Build the Cython files with coveragerc support to check unit test coverage.

    -\\-NTHREAD
      Specify the number of threads to allocate to compiling each module.
      This option is primarily useful for rapid development, particularly when all changes are in one module.
      While the -j option will not help parallelize this case, this option allows compilation of multiple source files belonging to the same module in parallel.

.. note::

    freud makes use of submodules. If you ever wish to manually update these, you can execute:

    .. code-block:: bash

        git submodule update --init

Unit Tests
==========

The unit tests for freud are included in the repository and are configured to be run using the Python :mod:`unittest` library:

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
