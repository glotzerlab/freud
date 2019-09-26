==========
Box Module
==========

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    freud.box.Box
    freud.box.PeriodicBuffer

.. rubric:: Details

.. automodule:: freud.box
    :synopsis: Represents periodic boxes.

Box
===

.. autoclass:: freud.box.Box(Lx=None, Ly=None, Lz=None, xy=None, xz=None, yz=None, is2D=None)
    :members: cube, from_box, from_matrix, getImage, getLatticeVector, is2D, makeCoordinates, makeFraction, square, to_dict, to_matrix, to_tuple, unwrap, wrap

Particle Buffer
===============

.. autoclass:: freud.box.PeriodicBuffer(box)
    :members: compute
