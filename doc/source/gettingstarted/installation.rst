.. _installation:

Installation
============

**freud** binaries are available on conda-forge_ and PyPI_. You can also compile **freud** from
source.

Binaries
--------

conda-forge package
^^^^^^^^^^^^^^^^^^^

**freud** is available on conda-forge_ for the *linux-64*, *osx-64*, *osx-arm64* and *win-64*
architectures. Execute one of the following commands to install **freud**:

.. code-block:: bash

   micromamba install freud

**OR**

.. code-block:: bash

   mamba install freud

PyPI
^^^^

Use **uv** or **pip** to install **freud** binaries from PyPI_ into a virtual environment:

.. code:: bash

   uv pip install freud-analysis

**OR**

.. code:: bash

   python3 -m pip install freud-analysis

.. _conda-forge: https://conda-forge.org/
.. _PyPI: https://pypi.org/

Compile from source
-------------------

To build **freud** from source:

1. `Obtain the source`_:

   .. code-block:: bash

      git clone --recursive git@github.com:glotzerlab/freud.git

2. Change to the repository directory::

    cd freud

3. `Install with uv`_::

    uv pip install .

4. **OR** `Install prerequisites`_ and `Build with CMake for development`_:

   .. code-block:: bash

      micromamba install cmake ninja numpy python tbb-devel nanobind scikit-build-core gsd matplotlib pytest rowan scipy sympy

   .. code-block:: bash
    
      cmake -B build -S . -GNinja
      cd build
      ninja

To run the tests:

1. `Run tests`_::

    cd tests
    PYTHONPATH=../build python3 -m pytest

To build the documentation from source:

1. `Install prerequisites`_::

    micromamba install sphinx furo nbsphinx jupyter_sphinx sphinxcontrib-bibtex sphinx-copybutton

2. `Build the documentation`_:

   .. code-block:: bash

      cd {{ path/to/freud/repository }}
   
   .. code-block:: bash

      sphinx-build -b html doc html

The sections below provide details on each of these steps.

.. _Install prerequisites:

Install prerequisites
^^^^^^^^^^^^^^^^^^^^^

**freud** requires a number of tools and libraries to build.

**General requirements:**

- A C++17-compliant compiler.
- **CMake** >= 3.15.0
- **Intel Threading Building Blocks** >= 2019.7
- **nanobind** >= 2.0.0
- **NumPy** >= 1.19.0
- **Python** >= 3.9
- **scikit-build-core**

**To execute unit tests:**

- **dynasor** (optional)
- **gsd**
- **matplotlib**
- **pytest**
- **rowan**
- **scipy**
- **sympy**

.. _Obtain the source:

Obtain the source
^^^^^^^^^^^^^^^^^

Clone using Git_:

.. code-block:: bash

   git clone --recursive git@github.com:glotzerlab/freud.git

Release tarballs are also available on the `GitHub release pages`_.

.. seealso::

    See the `git book`_ to learn how to work with `Git`_ repositories.

.. _GitHub release pages: https://github.com/glotzerlab/freud/releases/
.. _git book: https://git-scm.com/book
.. _Git: https://git-scm.com/


.. _Install with uv:

Install with uv
^^^^^^^^^^^^^^^^

Use **uv** to install the Python module into your virtual environment:

.. code-block:: bash

   cd {{ path/to/freud/repository }}

.. code-block:: bash

   uv pip install .

To perform incremental builds, `install the prerequisites first <Install prerequisites>`_, then run:

.. code-block:: bash
    
   uv pip install --no-deps --no-build-isolation --force-reinstall -C build-dir=$PWD/build .

You may find using `CMake`_ directly more effective for incremental builds (see the next section).

.. Build with CMake for development:

Build with CMake for development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**freud** also provides `CMake`_ scripts for development and testing that build a functional Python
module in the build directory. First, configure the build with ``cmake``:

.. code-block:: bash

   cd {{ path/to/freud/repository }}

.. code-block:: bash

   cmake -B build -S . -GNinja

Then, build the code:

.. code-block:: bash

   cd build
   ninja

Execute ``ninja`` to rebuild after you modify the code. ``ninja`` will automatically reconfigure
as needed.

.. tip::

    Pass the following options to ``cmake`` to optimize the build for your processor:
    ``-DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native``.

.. warning::

    When using a ``conda-forge`` environment for development, make sure that the environment does
    not contain ``clang``, ``gcc``, or any other compiler or linker. These interfere with the native
    compilers on your system and will result in compiler errors when building, linker errors when
    running, or segmentation faults.

.. _CMake: https://cmake.org/
.. _Ninja: https://ninja-build.org/


Run tests
^^^^^^^^^

.. note::

    You must first `Obtain the source`_ before you can run the tests.

Use `pytest`_ to execute unit tests:

.. code-block:: bash

    cd {{ path/to/freud/repository }}
    cd tests

.. code-block:: bash

   PYTHONPATH=../build python3 -m pytest

.. _pytest: https://docs.pytest.org/


.. _Build the documentation:

Build the documentation
^^^^^^^^^^^^^^^^^^^^^^^

Run `Sphinx`_ to build the HTML documentation:

.. code-block:: bash

   PYTHONPATH=build sphinx-build -b html doc/source html

Open the file :file:`html/index.html` in your web browser to view the documentation.

.. tip::

    Add the sphinx options ``-a -n -W -T --keep-going`` to produce docs with consistent links in
    the side panel and provide more useful error messages.

.. _Sphinx: https://www.sphinx-doc.org/

.. important::

    You must clone the freud GitHub repository to build the documentation.
