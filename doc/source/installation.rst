============
Installation
============

Requirements
============

- `NumPy <http://www.numpy.org/>`_ is **required** to build freud
- `Cython <http://cython.org/>`_ >= 0.23 is **required** to compile your own _freud.cpp file.
  * Cython **is not required** to install freud
- `Boost <http://www.boost.org/>`_ is **required** to run freud
- `Intel Thread Building Blocks <https://www.threadingbuildingblocks.org/>`_ is **required** to run freud

Documentation
=============

You may download the documentation in the `downloads section <https://bitbucket.org/glotzer/freud/downloads>`_, \
or you may build the documentation yourself:

Building the documentation
++++++++++++++++++++++++++

Documentation written in sphinx, not doxygen. Please install sphinx:

.. code-block:: bash

    $: conda install sphinx

OR

.. code-block:: bash

    $: pip install sphinx

To view the full documentation run the following commands in the source directory:

Linux
-----

.. code-block:: bash

    $: cd doc
    $: make html
    $: firefox build/html/index.html

Mac
---

.. code-block:: bash

    $: cd doc
    $: make html
    $: open build/html/index.html

If you have latex and/or pdflatex, you may also build a pdf of the documentation:

Linux
-----

.. code-block:: bash

    $: cd doc
    $: make latexpdf
    $: xdg-open build/latex/freud.pdf

Mac
---

.. code-block:: bash

    $: cd doc
    $: make latexpdf
    $: open build/latex/freud.pdf

Installation
============

Install freud via `conda <http://conda.pydata.org/docs/>`_, \
`glotzpkgs <http://glotzerlab.engin.umich.edu/glotzpkgs/>`_, or compile from source.

Conda install
+++++++++++++

To install freud with conda, make sure you have the glotzer channel and conda-private channels installed:

.. code-block:: bash

    $: conda config --add channels glotzer
    $: conda config --add channels file:///nfs/glotzer/software/conda-private

Now, install freud

.. code-block:: bash

    $: conda install freud
    # you may also install into a new environment
    $: conda create -n my_env python=3.5 freud

glotzpkgs install
+++++++++++++++++

*Please refer to the official `glotzpkgs <http://glotzerlab.engin.umich.edu/glotzpkgs/>`_ documentation*

*Make sure you have a working glotzpkgs env.*

.. code-block:: bash

    # install from provided binary
    $: gpacman -S freud
    # installing your own version
    $: cd /pth/to/glotzpkgs/freud
    $: gmakepkg
    # tab completion is your friend here
    $: gpacman -U freud-<version>-flux.pkg.tar.gz
    # now you can load the binary
    $: module load freud

Compile from source
+++++++++++++++++++

It's easiest to install freud with a working conda install of the required packages:

- python (2.7, 3.4, 3.5, 3.6)
- numpy
- boost (2.7, 3.3 provided on flux, 3.4, 3.5)
- cython (not required, but a correct _freud.cpp file must be present to compile)
- tbb
- cmake
- icu (because of boost for now)

You may either make a build directory *inside* the freud source directory, or create a separate build directory somewhere on your system:

.. code-block:: bash

    $: mkdir /pth/to/build
    $: cd /pth/to/build
    $: ccmake /pth/to/freud
    # adjust settings as needed, esp. ENABLE_CYTHON=ON
    $: make install -j6
    # enjoy

By default, freud installs to the `USER_SITE <https://docs.python.org/2/install/index.html>`_ directory. Which is in \
`~/.local` on linux and in `~/Library` on mac. `USER_SITE` is on the python search path by default, there is no need \
to modify `PYTHONPATH`.

To run out of the build directory, run `make -j20` instead of `make install -j20` and then add the build directory to your `PYTHONPATH`:

.. note::

    freud makes use of submodules. CMake has been configured to automatically init and update submodules. However, if
    this does not work, or you would like to do this yourself, please execute:

    .. code-block:: bash

        git submodule update --init

Unit Tests
==========

Run all unit tests with nosetests in the source directory. To add a test, simply add a file to the `tests` directory, \
and nosetests will automatically discover it. See http://pythontesting.net/framework/nose/nose-introduction/ for an \
introduction to writing nose tests.

.. code-block:: bash

    # Install nose if necessary
    $: conda install nose
    # run tests
    $: cd source
    $: nosetests
