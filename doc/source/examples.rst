========
Examples
========

Examples are provided as `Jupyter <https://jupyter.org/>`_ notebooks in a separate
`freud-examples <https://github.com/glotzerlab/freud-examples>`_ repository.
These notebooks may be launched `interactively on Binder <https://mybinder.org/v2/gh/glotzerlab/freud-examples/master?filepath=index.ipynb>`_
or downloaded and run on your own system.
Visualization of data is done via `Matplotlib <https://matplotlib.org/>`_ [Matplotlib]_ and `Bokeh <https://bokeh.pydata.org/>`_ [Bokeh]_.


Key concepts
============

There are a few critical concepts, algorithms, and data structures that are central to all of `freud`.
The `box` module defines the concept of a periodic simulation box, and the `locality` module defines methods for finding nearest neighbors for particles.
Since both of these are used throughout `freud`, we recommend familiarizing yourself with these before delving too deep into the workings of specific `freud` modules.

.. toctree::
    :maxdepth: 1
    :caption: Examples
    :glob:

    examples/module_intros/Box*
    examples/module_intros/Locality*

Module Examples
===============

The remaining examples go into greater detail on the functionality of specific modules, showing how they can be used to perform specific types of analyses of simulations.

.. toctree::
    :maxdepth: 1
    :caption: Examples
    :glob:

    examples/module_intros/Cluster*
    examples/module_intros/Density*
    examples/module_intros/Environment*
    examples/module_intros/Interface*
    examples/module_intros/Order*
    examples/module_intros/PMFT*
    examples/module_intros/Voronoi*
