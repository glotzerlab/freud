===================
freud documentation
===================

*"Neurosis is the inability to tolerate ambiguity" - Sigmund Freud*

Welcome to the documentation for freud, a Python package for analyzing particle simulation trajectories of periodic systems.
The library contains a diverse array of analysis routines designed for molecular dynamics and Monte Carlo.
Since any scientific investigation is likely to benefit from a range of analyses, freud is designed to work as part of a larger analysis pipeline.
In order to maximize its interoperabiity with other systems, freud works with and returns `NumPy <http://www.numpy.org/>`_ arrays.

Installing freud
================

The recommended method of installing freud is using [conda](https://conda.io/docs/) through the [conda-forge channel](https://conda-forge.org/).
First download and install [miniconda](https://conda.io/miniconda.html) following [conda's instructions](https://conda.io/docs/user-guide/install/index.html).
Then, install freud from conda-forge:
  
.. code-block:: bash

    $ conda install -c conda-forge freud

Alternatively, freud can be installed directly from source.

.. code-block:: bash

    $ mkdir build
    $ cd build
    $ cmake ../
    $ make install

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
Any issues or bugs may be reported at our `issue tracker <https://bitbucket.org/glotzer/freud/issues?status=new&status=open>`_, while questions and discussion can be directed to our `forum <https://groups.google.com/forum/#!forum/freud-users>`_.
All contributions to freud are welcomed via pull requests!
Please see the :doc:`development guide <development>` for more information on requirements for new code.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
