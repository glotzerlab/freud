# <img src="https://raw.githubusercontent.com/glotzerlab/freud/master/doc/source/images/freud_favicon.png" width="75" height="75"> freud

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.166564.svg)](https://doi.org/10.5281/zenodo.166564)
[![PyPI](https://img.shields.io/pypi/v/freud-analysis.svg)](https://pypi.org/project/freud-analysis/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/freud.svg)](https://anaconda.org/conda-forge/freud)
[![ReadTheDocs](https://readthedocs.org/projects/freud/badge/?version=latest)](https://freud.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/glotzerlab/freud-examples/master?filepath=index.ipynb)
[![Codecov](https://codecov.io/gh/glotzerlab/freud/branch/master/graph/badge.svg)](https://codecov.io/gh/glotzerlab/freud)
[![GitHub Stars](https://img.shields.io/github/stars/glotzerlab/freud.svg?style=social)](https://github.com/glotzerlab/freud)

The *freud* Python library provides a simple, flexible, powerful set of tools for analyzing trajectories obtained from molecular dynamics or Monte Carlo simulations.
High performance, parallelized C++ is used to compute standard tools such as radial distribution functions, correlation functions, and clusters, as well as original analysis methods including potentials of mean force and torque (PMFTs) and local environment matching.
The *freud* library uses [NumPy arrays](https://www.numpy.org/) for input and output, enabling integration with the scientific Python ecosystem for many typical materials science workflows.

When using *freud* to process data for publication, please [use this citation](https://doi.org/10.5281/zenodo.166564).

## Resources

- [*freud* Documentation](https://freud.readthedocs.io/):
  Python package reference.
  - [Installation Guide](https://freud.readthedocs.io/en/stable/installation.html):
    Instructions for installing and compiling *freud*.
  - [Examples](https://freud.readthedocs.io/en/stable/examples.html):
    Jupyter notebooks demonstrating the core features of *freud*.
  - [API Reference](https://freud.readthedocs.io/en/stable/modules.html):
    Documentation for classes and methods in *freud*.
  - [Development Guide](https://freud.readthedocs.io/en/stable/development.html):
    Resources for contributing to *freud*.
- [freud-users Google Group](https://groups.google.com/d/forum/freud-users):
  Ask questions to the *freud* community.
- [Issue Tracker](https://github.com/glotzerlab/freud/issues):
  Report issues and suggest feature enhancements.

## Installation

Install via conda:

```bash
conda install -c conda-forge freud
```

Or via pip:
```bash
pip install freud-analysis
```

*freud* is also available via containers for [Docker](https://hub.docker.com/r/glotzerlab/software) or [Singularity](https://singularity-hub.org/collections/1663).

Please refer to the [Installation Guide](https://freud.readthedocs.io/en/stable/installation.html) to compile freud from source.

## Examples

The *freud* library is called using Python scripts.
Many core features are [demonstrated in the *freud* documentation](https://freud.readthedocs.io/en/stable/examples.html).
Additional example Jupyter notebooks can be found in the [*freud* examples repository](https://github.com/glotzerlab/freud-examples).
These notebooks may be launched [interactively on Binder](https://mybinder.org/v2/gh/glotzerlab/freud-examples/master?filepath=index.ipynb)
or downloaded and run on your own system.
Below is a script that computes the radial distribution function.

```python
import freud

# Create a freud compute object (rdf is the canonical example)
rdf = freud.density.RDF(rmax=5, dr=0.1)

# Load in your data (freud does not provide a data reader)
box_data = np.load("path/to/box_data.npy")
pos_data = np.load("path/to/pos_data.npy")

# Create freud box
box = freud.box.Box(Lx=box_data[0]["Lx"], Ly=box_data[0]["Ly"], is2D=True)

# Compute RDF
rdf.compute(box, pos_data[0], pos_data[0])
# Get bin centers, RDF data
r = rdf.R
y = rdf.RDF
```
