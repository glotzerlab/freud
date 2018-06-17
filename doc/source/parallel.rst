===============
Parallel Module
===============

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    freud.parallel.NumThreads
    freud.parallel.setNumThreads

.. rubric:: Details

The :py:class:`freud.parallel` module controls the parallelization behavior of freud, determining how many threads the TBB-enabled parts of freud will use.
By default, freud tries to use all available threads for parallelization unless directed otherwise, with one exception.

.. automodule:: freud.parallel
    :members:

    .. automethod:: freud.parallel.setNumThreads
