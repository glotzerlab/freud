==========
Box Module
==========

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    freud.box.Box
    freud.box.ParticleBuffer

.. rubric:: Details

.. automodule:: freud.box
    :synopsis: Represents periodic boxes.

Box
===

.. autoclass:: freud.box.Box(Lx, Ly, Lz, xy, xz, yz, is2D=None)
    :members: cube, from_box, from_matrix, getCoordinates, getImage, getLatticeVector, is2D, makeFraction, square, to_dict, to_matrix, to_tuple, unwrap, wrap

.. autoclass:: freud.box.ParticleBuffer(box)
    :members: compute
