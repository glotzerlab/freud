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

.. autoclass:: freud.box.Box(Lx, Ly, Lz=0, xy=0, xz=0, yz=0, is2D=None)
    :members: cube, from_box, from_matrix, get_images, get_lattice_vector, make_absolute, make_fractional, square, to_dict, to_matrix, unwrap, wrap

Periodic Buffer
===============

.. autoclass:: freud.box.PeriodicBuffer(box)
    :members: compute
