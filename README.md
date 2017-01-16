# Freud

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.166564.svg)](https://doi.org/10.5281/zenodo.166564)

Please use the above citation when using Freud.

Welcome to Freud! Freud provides users the ability to analyze generic data from a variety of sources including
simulation and experimental data for advanced metrics such as the radial distribution function and various order parameters.

## Mailing List

If you have a question, please consider posting to the [Freud-Users mailing list](https://groups.google.com/forum/#!forum/freud-users).

## Examples

Example Jupyter notebooks can be found in a [separate repository](https://bitbucket.org/glotzer/freud-examples). These
examples are available as a static notebook on [nbviewer](http://nbviewer.jupyter.org/github/harperic/freud-examples/blob/master/index.ipynb)
and as an interactive version on [mybinder](http://mybinder.org:/repo/harperic/freud-examples) in the near future.

## Installing Freud

Official binaries of Freud are available via [conda](http://conda.pydata.org/docs/) through
the [glotzer channel](https://anaconda.org/glotzer).
To install Freud, first download and install
[miniconda](http://conda.pydata.org/miniconda.html) following [conda's instructions](http://conda.pydata.org/docs/install/quick.html).
Then add the `glotzer` channel and install Freud:

```bash
$ conda config --add channels glotzer
$ conda install freud
```

## Compiling Freud

Use cmake to configure an out of source build and make to build freud.

```bash
mkdir build
cd build
cmake ../
make -j20
```

By default, freud installs to the [USER_SITE](https://docs.python.org/2/install/index.html) directory. Which is in
`~/.local` on linux and in `~/Library` on mac. `USER_SITE` is on the python search path by default, there is no need to
modify `PYTHONPATH`.

To run out of the build directory, add the build directory to your `PYTHONPATH`:

~~~
bash
export PYTHONPATH=`pwd`:$PYTHONPATH
~~~

For more detailed instructions, [see the documentation](http://glotzerlab.engin.umich.edu/freud/).

#### Note

Freud makes use of submodules. CMAKE has been configured to automatically init and update submodules. However, if
this does not work, or you would like to do this yourself, please execute:

```bash
git submodule init
```

### Requirements

* Required:
    * Python >= 2.7 (3.x recommended)
    * Numpy >=1.7
    * Boost (headers only)
    * CMake >= 2.8.0 (to compile freud)
    * C++ 11 capable compiler (tested with gcc >= 4.8.5, clang 3.5)
    * Intel Thread Building Blocks
* Optional:
    * Cython >= 0.23 (to compile your own _freud.cpp)

## Job scripts

Freud analysis scripts are python scripts.

Here is a simple example.

```python
import freud

# create a freud compute object (rdf is the canonical example)
rdf = freud.density.rdf(rmax=5, dr=0.1)
# load in your data (freud does not provide a data reader)
box_data = np.load("pth/to/box_data.npy")
pos_data = np.load("pth/to/pos_data.npy")

# create freud box
box = freud.box.Box(Lx=box_data[0]["Lx"], Ly=box_data[0]["Ly"], is2D=True)
# compute RDF
rdf.compute(box, pos_data[0], pos_data[0])
# get bin centers, rdf data
r = rdf.getR()
y = rdf.getRDF()
```

## Documentation

You may [read the documentation online](http://glotzerlab.engin.umich.edu/freud/), download the
documentation in the [downloads section](https://bitbucket.org/glotzer/freud/downloads), or you may build the
documentation yourself:

### Building the documentation

Documentation written in sphinx. Please install sphinx:

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

Run all unit tests with nosetests in the source directory. To add a test, simply add a file to the `tests` directory,
and nosetests will automatically discover it. See http://pythontesting.net/framework/nose/nose-introduction/ for
an introduction to writing nose tests.

~~~
cd source
nosetests
~~~
