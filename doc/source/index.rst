===================
freud documentation
===================

|DOI|
|Anaconda-Server|
|Binder|
|ReadTheDocs|
|Codecov|

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.166564.svg
    :target: https://doi.org/10.5281/zenodo.166564
.. |Anaconda-Server| image:: https://anaconda.org/conda-forge/freud/badges/version.svg
    :target: https://anaconda.org/conda-forge/freud
.. |Binder| image:: https://mybinder.org/badge.svg
    :target: https://mybinder.org:/repo/harperic/freud-examples
.. |ReadTheDocs| image:: https://readthedocs.org/projects/freud/badge/?version=latest
    :target: https://freud.readthedocs.io/en/latest/?badge=latest
.. |Codecov| image:: https://codecov.io/bb/glotzer/freud/branch/master/graph/badge.svg
    :target: https://codecov.io/bb/glotzer/freud

*"Neurosis is the inability to tolerate ambiguity" - Sigmund Freud*

Welcome to the documentation for freud, a Python package for analyzing particle simulation trajectories of periodic systems.
The library contains a diverse array of analysis routines designed for molecular dynamics and Monte Carlo simulation trajectories.
Since any scientific investigation is likely to benefit from a range of analyses, freud is designed to work as part of a larger analysis pipeline.
In order to maximize its interoperability with other systems, freud works with and returns `NumPy <http://www.numpy.org/>`_ arrays.

Installing freud
================

The recommended method of installing freud is using `conda <https://conda.io/docs/>`_ through the `conda-forge channel <https://conda-forge.org/>`_.
First download and install `miniconda <https://conda.io/miniconda.html>`_ following `conda's instructions <https://conda.io/docs/user-guide/install/index.html>`_.
Then, install freud from conda-forge:
  
.. code-block:: bash

    $ conda install -c conda-forge freud

Alternatively, freud can be installed directly from source.

.. code-block:: bash

    $ git clone https://bitbucket.org/glotzer/freud.git
    $ cd freud
    $ python setup.py install --user

.. toctree::
   :maxdepth: 2
   :caption: Contents

   examples
   installation
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
