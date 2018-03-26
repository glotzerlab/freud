# freud

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.166564.svg)](https://doi.org/10.5281/zenodo.166564)
[![Anaconda-Server Badge](https://anaconda.org/glotzer/freud/badges/version.svg)](https://anaconda.org/glotzer/freud)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org:/repo/harperic/freud-examples)
[![ReadTheDocs](https://readthedocs.org/projects/freud/badge/?version=latest)](https://freud.readthedocs.io/en/latest/?badge=latest)

The freud library provides users the ability to analyze molecular dynamics and Monte Carlo simulation trajectories for advanced metrics such as the radial distribution function and various order parameters. Its modules work with and return NumPy arrays, and are able to process both 2D and 3D data. Features in freud include computing the radial distribution function, local density, hexagonal order parameter and local bond order parameters, Voronoi tessellations, k-space quantities, and more.

When using freud to process data for publication, please [use this citation](https://doi.org/10.5281/zenodo.166564).

## Mailing List

If you have a question, please consider posting to the
[freud-users mailing list](https://groups.google.com/forum/#!forum/freud-users).

## Examples

Example Jupyter notebooks can be found in a [separate repository](https://bitbucket.org/glotzer/freud-examples). These examples are available as a static notebook on [nbviewer](http://nbviewer.jupyter.org/github/harperic/freud-examples/blob/master/index.ipynb) and as an interactive version on [mybinder](http://mybinder.org:/repo/harperic/freud-examples).

## Installing freud

Official binaries of freud are available via [conda](https://conda.io/docs/) through the [glotzer channel](https://anaconda.org/glotzer). To install freud, first download and install [miniconda](https://conda.io/miniconda.html) following [conda's instructions](https://conda.io/docs/user-guide/install/index.html). Then add the `glotzer` channel and install freud:

```bash
$ conda config --add channels glotzer
$ conda install freud
```

## Compiling freud

Use CMake to configure and make freud from source.

```bash
mkdir build
cd build
cmake ../
make -j20
```

By default, freud installs to the [USER_SITE](https://docs.python.org/2/install/index.html) directory. Which is in
`~/.local` on Linux and in `~/Library` on macOS. `USER_SITE` is on the Python search path by default, so there is no need to
modify `PYTHONPATH`.

To run out of the build directory, add the build directory to your `PYTHONPATH`:

~~~
bash
export PYTHONPATH=`pwd`:$PYTHONPATH
~~~

For more detailed instructions, see [the documentation](https://freud.readthedocs.io).

#### Note

The freud library makes use of submodules. CMake has been configured to automatically init and update submodules.
However, if this does not work, or you would like to do this yourself, please execute:

```bash
git submodule update --init
```

### Requirements

* Required:
    * Python >= 2.7 (3.5+ recommended)
    * NumPy >= 1.7
    * Boost (headers only)
    * CMake >= 2.8.0 (to compile freud)
    * C++ 11 capable compiler (tested with gcc >= 4.8.5, clang 3.5)
    * Intel Threading Building Blocks
* Optional:
    * Cython >= 0.23 (to compile your own _freud.cpp)

## Job scripts

The freud library is called using Python scripts.

Here is a simple example.

```python
import freud

# create a freud compute object (rdf is the canonical example)
rdf = freud.density.rdf(rmax=5, dr=0.1)
# load in your data (freud does not provide a data reader)
box_data = np.load("path/to/box_data.npy")
pos_data = np.load("path/to/pos_data.npy")

# create freud box
box = freud.box.Box(Lx=box_data[0]["Lx"], Ly=box_data[0]["Ly"], is2D=True)
# compute RDF
rdf.compute(box, pos_data[0], pos_data[0])
# get bin centers, rdf data
r = rdf.getR()
y = rdf.getRDF()
```

## Documentation

The documentation is available online at [https://freud.readthedocs.io](https://freud.readthedocs.io).

To build the documentation yourself, please install sphinx:

	conda install sphinx

OR

	pip install sphinx

To view the full documentation run the following commands in the source directory:

~~~bash
# Linux
cd doc
make html
xdg-open build/html/index.html

# Mac
cd doc
make html
open build/html/index.html
~~~

If you have latex and/or pdflatex, you may also build a pdf of the documentation:

~~~bash
# Linux
cd doc
make latexpdf
xdg-open build/latex/freud.pdf

# Mac
cd doc
make latexpdf
open build/latex/freud.pdf
~~~

## Unit Tests

Run all unit tests with `nosetests .` in the `tests` directory. To add a test, simply add a file to the `tests` directory, and nosetests will automatically discover it.
Refer to this [introduction to nose](http://pythontesting.net/framework/nose/nose-introduction/) for help writing tests.

~~~
cd tests
nosetests .
~~~
