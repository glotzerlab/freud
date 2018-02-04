============
Installation
============

Requirements
============

- `NumPy <http://www.numpy.org/>`_ is **required** to build freud
- `Cython <http://cython.org/>`_ >= 0.23 is **required** to compile your own :code:`_freud.cpp` file. Cython **is not required** to install freud
- `Boost <http://www.boost.org/>`_ is **required** to run freud
- `Intel Threading Building Blocks <https://www.threadingbuildingblocks.org/>`_ is **required** to run freud

Documentation
=============

You may use the online documentation from `ReadTheDocs <https://freud.readthedocs.io/>`_, \
or you may build the documentation yourself:

Building the documentation
++++++++++++++++++++++++++

The documentation is build with sphinx. To install sphinx, run

.. code-block:: bash

    conda install sphinx

or

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


Installation
============

Install freud via `conda <http://conda.pydata.org/docs/>`_, \
`glotzpkgs <http://glotzerlab.engin.umich.edu/glotzpkgs/>`_, or compile from source.

Conda install
+++++++++++++

The code below will enable the glotzer conda channel and install freud.

.. code-block:: bash

    conda config --add channels glotzer
    conda install freud

glotzpkgs install
+++++++++++++++++

Please refer to the official `glotzpkgs <http://glotzerlab.engin.umich.edu/glotzpkgs/>`_ documentation.

First, make sure you have a working glotzpkgs environment.

.. code-block:: bash

    # install from provided binary
    gpacman -S freud
    # installing your own version
    cd /path/to/glotzpkgs/freud
    gmakepkg
    # tab completion is your friend here
    gpacman -U freud-<version>-flux.pkg.tar.gz
    # now you can load the binary
    module load freud

Compile from source
+++++++++++++++++++

It's easiest to install freud with a working conda install of the required packages:

- python (2.7, 3.4, 3.5, 3.6)
- numpy
- boost (2.7, 3.3 provided on flux, 3.4, 3.5)
- icu (requirement of boost)
- cython (not required, but a correct :code:`_freud.cpp` file must be present to compile)
- tbb
- cmake



The code that follows creates a build directory inside the freud source directory and builds freud:

.. code-block:: bash

    mkdir build
    cd build
    cmake ../
    # Use `cmake ../ -DENABLE_CYTHON=ON` to rebuild _freud.cpp
    make install -j4

By default, freud installs to the `USER_SITE <https://docs.python.org/3.6/library/site.html#site.USER_SITE>`_ directory.
:code:`USER_SITE` is on the Python search path by default, so there is no need to modify :code:`PYTHONPATH`.

To run out of the build directory, run :code:`make -j4` instead of :code:`make install -j4` and then add the build directory to
your :code:`PYTHONPATH`.

.. note::

    freud makes use of submodules. CMake has been configured to automatically init and update submodules. However, if
    this does not work, or you would like to do this yourself, please execute:

    .. code-block:: bash

        git submodule update --init

Unit Tests
==========

Run all unit tests with :code:`nosetests` in the source directory. To add a test, simply add a file to the `tests` directory, and nosetests will automatically discover it. Read this `introduction to nosetests <http://pythontesting.net/framework/nose/nose-introduction/>`_ for more information.

.. code-block:: bash

    # Install nose
    conda install nose
    # Run tests from the source directory
    nosetests
