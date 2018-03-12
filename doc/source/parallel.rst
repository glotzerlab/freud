===============
Parallel Module
===============

The :py:class:`freud.parallel` module tries to use all available threads for
parallelization unless directed otherwise, with one exception. On the *flux*
and *nyx* clusters, freud will only use one core unless directed otherwise.

.. automodule:: freud.parallel
    :members:

    .. automethod:: freud.parallel.setNumThreads
