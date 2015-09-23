.. contents:: Freud modules

==============
Density Module
==============

Lorem Ipsum

Available functions
===================

.. automodule:: freud.density

    .. autoclass:: GaussianDensity
        :members:

        .. method:: __init__(self, width, r_cut, dr)

        Initialize with all dimensions identical

        .. method:: __init__(self, width_x, width_y, width_z, r_cut, dr)

        Initialize with specific dimensions

    .. autoclass:: RDF
        :members:

        .. method:: __init__(self, rmax, dr)

    .. autoclass:: FloatCF
        :members:

        .. method:: __init__(self, rmax, dr)

    .. autoclass:: ComplexCF
        :members:

        .. method:: __init__(self, rmax, dr)
