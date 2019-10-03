=====
freud
=====

|DOI|
|PyPI|
|conda-forge|
|ReadTheDocs|
|Binder|
|Codecov|
|GitHub-Stars|

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.166564.svg
   :target: https://doi.org/10.5281/zenodo.166564
.. |PyPI| image:: https://img.shields.io/pypi/v/freud-analysis.svg
   :target: https://pypi.org/project/freud-analysis/
.. |conda-forge| image:: https://img.shields.io/conda/vn/conda-forge/freud.svg
   :target: https://anaconda.org/conda-forge/freud
.. |ReadTheDocs| image:: https://readthedocs.org/projects/freud/badge/?version=latest
   :target: https://freud.readthedocs.io/en/latest/?badge=latest
.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/glotzerlab/freud-examples/master?filepath=index.ipynb
.. |Codecov| image:: https://codecov.io/gh/glotzerlab/freud/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/glotzerlab/freud
.. |GitHub-Stars| image:: https://img.shields.io/github/stars/glotzerlab/freud.svg?style=social
   :target: https://github.com/glotzerlab/freud

.. danger::
   **freud v2.0 is currently in beta on the** :code:`master` **branch.**
   This major rewrite of the freud library brings new APIs, speedups,
   and many new features. Please
   `report any issues <https://github.com/glotzerlab/freud/issues>`_
   you encounter while using the beta or learning the new API.


Overview
========

The **freud** Python library provides a simple, flexible, powerful set of
tools for analyzing trajectories obtained from molecular dynamics or
Monte Carlo simulations. High performance, parallelized C++ is used to
compute standard tools such as radial distribution functions,
correlation functions, and clusters, as well as original analysis
methods including potentials of mean force and torque (PMFTs) and local
environment matching. The **freud** library uses `NumPy
arrays <https://www.numpy.org/>`__ for input and output, enabling
integration with the scientific Python ecosystem for many typical
materials science workflows.

When using **freud** to process data for publication, please `use this
citation <https://doi.org/10.5281/zenodo.166564>`__.


Installation
============

The easiest ways to install **freud** are using pip:

.. code:: bash

   pip install freud-analysis

or conda:

.. code:: bash

   conda install -c conda-forge freud

**freud** is also available via containers for `Docker
<https://hub.docker.com/r/glotzerlab/software>`_ or `Singularity
<https://singularity-hub.org/collections/1663>`_.  If you need more detailed
information or wish to install **freud** from source, please refer to the
`Installation Guide
<https://freud.readthedocs.io/en/stable/installation.html>`_ to compile freud
from source.


Resources
=========

Some other helpful links for working with freud:

-  Find examples of using **freud** on the `examples page <https://freud.readthedocs.io/en/stable/examples.html>`_.
-  Find detailed tutorials and reference information at the official `freud documentation <https://freud.readthedocs.io/>`_.
-  View and download the code on the `GitHub repository <https://github.com/glotzerlab/freud>`_.
-  Ask for help on the `freud-users Google Group <https://groups.google.com/d/forum/freud-users>`_.
-  Report issues or request features using the `issue tracker <https://github.com/glotzerlab/freud/issues>`_.


Examples
========

The **freud** library is called using Python scripts. Many core features are
`demonstrated in the freud documentation
<https://freud.readthedocs.io/en/stable/examples.html>`_. The examples come in
the form of Jupyter notebooks, which can also be downloaded from the `freud
examples repository <https://github.com/glotzerlab/freud-examples>`_ or
`launched interactively on Binder
<https://mybinder.org/v2/gh/glotzerlab/freud-examples/master?filepath=index.ipynb>`_.
Below is a script that computes the radial distribution function.

.. code:: python

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


Support and Contribution
========================

Please visit our repository on `GitHub <https://github.com/glotzerlab/freud>`_ for the library source code.
Any issues or bugs may be reported at our `issue tracker <https://github.com/glotzerlab/freud/issues>`_, while questions and discussion can be directed to our `forum <https://groups.google.com/forum/#!forum/freud-users>`_.
All contributions to freud are welcomed via pull requests!
