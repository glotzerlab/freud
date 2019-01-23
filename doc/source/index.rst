===================
freud documentation
===================

|DOI|
|PyPI|
|Anaconda-Server|
|Binder|
|ReadTheDocs|
|Codecov|
|GitHub-Stars|

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.166564.svg
    :target: https://doi.org/10.5281/zenodo.166564
.. |PyPI| image:: https://img.shields.io/pypi/v/freud-analysis.svg
    :target: https://pypi.org/project/freud-analysis/
.. |Anaconda-Server| image:: https://anaconda.org/conda-forge/freud/badges/version.svg
    :target: https://anaconda.org/conda-forge/freud
.. |Binder| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/glotzerlab/freud-examples/master?filepath=index.ipynb
.. |ReadTheDocs| image:: https://readthedocs.org/projects/freud/badge/?version=latest
    :target: https://freud.readthedocs.io/en/latest/?badge=latest
.. |Codecov| image:: https://codecov.io/bb/glotzer/freud/branch/master/graph/badge.svg
    :target: https://codecov.io/bb/glotzer/freud
.. |GitHub-Stars| image:: https://img.shields.io/github/stars/glotzerlab/freud.svg?style=social
    :target: https://github.com/glotzerlab/freud

Overview
========

The freud Python library provides a simple, flexible, powerful set of tools for analyzing trajectories obtained from molecular dynamics or Monte Carlo simulations. High performance, parallelized C++ is used to compute standard tools such as radial distribution functions, correlation functions, and clusters, as well as original analysis methods including potentials of mean force and torque (PMFTs) and local environment matching. The freud library uses `NumPy arrays <http://www.numpy.org/>`_ for input and output, enabling integration with the scientific Python ecosystem for many typical materials science workflows.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   examples
   modules
   development
   citations
   license
   credits


Support and Contribution
========================

Please visit our repository on `Bitbucket <https://bitbucket.org/glotzer/freud>`_ for the library source code.
Any issues or bugs may be reported at our `issue tracker <https://bitbucket.org/glotzer/freud/issues>`_, while questions and discussion can be directed to our `forum <https://groups.google.com/forum/#!forum/freud-users>`_.
All contributions to freud are welcomed via pull requests!
Please see the :doc:`development guide <development>` for more information on requirements for new code.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
