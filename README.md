freud {#mainpage}
===============================================

# Requirements {#requirements_section}

Numpy is **required** to build freud

Cython >= 0.23 is **required** to compile your own _freud.cpp file. Cython **is not required** to install freud

Boost is **required** to run freud

Intel Thread Building Blocks is **require** to run freud

# Documentation {#documentation_section}

You may download the documentation in the [downloads section](https://bitbucket.org/glotzer/freud/downloads), or you may build the documentation yourself:

## Building the documentation

Documentation written in sphinx, not doxygen. Please install sphinx:

	conda install sphinx

OR

	pip install sphinx

To view the full documentation run the following commands in the source directory:

~~~
# Linux
cd doc
make html
firefox build/html/index.html

# Mac
cd doc
make html
open build/html/index.html
~~~

If you have latex and/or pdflatex, you may also build a pdf of the documentation:

~~~
# Linux
cd doc
make latexpdf
xdg-open build/latex/freud.pdf

# Mac
cd doc
make latexpdf
open build/latex/freud.pdf
~~~

# Installation

Freud may be installed via Conda, glotzpkgs, or compiled from source

## Conda install {#conda_section}

To install freud with conda, make sure you have the glotzer channel and conda-private channels installed:

~~~
$: conda config --add channels glotzer
$: conda config --add channels file:///nfs/glotzer/software/conda-private
~~~

Now, install freud

~~~
conda install freud
# you may also install into a new environment
conda create -n my_env python=3.5 freud
~~~

## glotzpkgs install {#glotzpkgs_section}

*Please refer to the official glotzpkgs documentation*

*Make sure you have a working glotzpkgs env.*

~~~
# install from provided binary
$: gpacman -S freud
# installing your own version
$: cd /pth/to/glotzpkgs/freud
$: gmakepkg
# tab completion is your friend here
$: gpacman -U freud-<version>-flux.pkg.tar.gz
# now you can load the binary
$: module load freud
~~~

## Compile from source {#build_section}

It's easiest to install freud with a working conda install of the required packages:

* python (2.7, 3.3, 3.4, 3.5)
* numpy
* boost (2.7, 3.3 provided on flux, 3.4, 3.5)
* cython (not required, but a correct _freud.cpp file must be present to compile)
* tbb
* cmake
* icu (because of boost for now)

You may either make a build directory *inside* the freud source directory, or create a separate build directory somewhere on your system:

~~~
mkdir /pth/to/build
cd /pth/to/build
ccmake /pth/to/freud
# adjust settings as needed, esp. ENABLE_CYTHON=ON
make install -j6
# enjoy
~~~

By default, freud installs to the [USER_SITE](https://docs.python.org/2/install/index.html) directory. Which is in
`~/.local` on linux and in `~/Library` on mac. `USER_SITE` is on the python search path by default, there is no need to
modify `PYTHONPATH`.

# Tests {#tests_section}

Run all unit tests with nosetests in the source directory. To add a test, simply add a file to the `tests` directory,
and nosetests will automatically discover it. See http://pythontesting.net/framework/nose/nose-introduction/ for
an introduction to writing nose tests.

~~~
cd source
nosetests
~~~
