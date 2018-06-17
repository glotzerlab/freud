==========
Box Module
==========

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    freud.box.Box

.. rubric:: Details

The box module provides the Box class, which defines the geometry of the simulation box.
The module natively supports periodicity by providing the fundamental features for wrapping vectors outside the box back into it.

.. autoclass:: freud.box.Box(Lx, Ly, Lz, xy, xz, yz, is2D=None)
    :members:
    :inherited-members:
