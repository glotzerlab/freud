# freud

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.166564.svg)](https://doi.org/10.5281/zenodo.166564)
[![PyPI](https://img.shields.io/pypi/v/freud-analysis.svg)](https://pypi.org/project/freud-analysis/)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/freud/badges/version.svg)](https://anaconda.org/conda-forge/freud)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/glotzerlab/freud-examples/master?filepath=index.ipynb)
[![ReadTheDocs](https://readthedocs.org/projects/freud/badge/?version=latest)](https://freud.readthedocs.io/en/latest/?badge=latest)
[![Codecov](https://codecov.io/bb/glotzer/freud/branch/master/graph/badge.svg)](https://codecov.io/bb/glotzer/freud)

The freud library provides users the ability to analyze molecular dynamics and Monte Carlo simulation trajectories
for advanced metrics such as the radial distribution function and various order parameters. Its modules work with
and return NumPy arrays, and are able to process both 2D and 3D data. Features in freud include computing the radial
distribution function, local density, hexagonal order parameter and local bond order parameters,
potentials of mean force and torque (PMFTs), Voronoi tessellations, and more.

When using freud to process data for publication, please [use this citation](https://doi.org/10.5281/zenodo.166564).

# freud Community

If you have a question, please post to the
[freud-users mailing list](https://groups.google.com/forum/#!forum/freud-users).
Please report issues and suggest feature enhancements via the [Bitbucket issues page](https://bitbucket.org/glotzer/freud/issues?status=new&status=open).

# Documentation

The documentation is available online at [https://freud.readthedocs.io](https://freud.readthedocs.io).
These pages include an installation guide, examples demonstrating many of freud's core modules, API reference, and development guides for adding new features.

# Examples
Many core features are [demonstrated in the freud documentation](https://freud.readthedocs.io/en/stable/examples.html).
Additional example Jupyter notebooks can be found in the [freud-examples repository](https://bitbucket.org/glotzer/freud-examples).
These notebooks may be launched [interactively on Binder](https://mybinder.org/v2/gh/glotzerlab/freud-examples/master?filepath=index.ipynb)
or downloaded and run on your own system.

# Installing freud

## Install via conda

```bash
conda install -c conda-forge freud
```

## Install via pip

```bash
pip install freud-analysis
```

## Compiling freud
Please refer to the [installation documentation](https://freud.readthedocs.io/en/stable/installation.html) for help compiling freud from source.

# Simple example script

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
r = rdf.R
y = rdf.RDF
```
