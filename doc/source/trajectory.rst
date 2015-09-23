.. contents:: Freud modules

=================
Trajectory Module
=================

Lorem Ipsum


Available functions
===================

.. automodule:: freud.trajectory

    .. autoclass:: Box
        :members:

        .. method:: __init__(self)

        null constructor

        .. method:: __init__(self, L)

        initializes cubic box of side length L

        .. method:: __init__(self, L, is2D)

        initializes box of side length L (will create a 2D/3D box based on is2D)

        .. method:: __init__(self, L=L, is2D=False)

        initializes box of side length L (will create a 2D/3D box based on is2D)

        .. method:: __init__(self, Lx, Ly, Lz)

        initializes cubic box of side length L

        .. method:: __init__(self, Lx, Ly, Lz, is2D=False)

        initializes box with side lengths Lx, Ly (, Lz if is2D=False)

        .. method:: __init__(self, Lx=0.0, Ly=0.0, Lz=0.0, xy=0.0, xz=0.0, yz=0.0, is2D=False)

        Preferred method to initialize. Pass in as kwargs. Any not set will be set to the listed defaults.
