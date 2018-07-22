# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

R"""
The index module exposes the :math:`1`-dimensional indexer utilized in freud at
the C++ level.  At the C++ level, freud utilizes flat arrays to represent
multidimensional arrays. :math:`N`-dimensional arrays with :math:`n_i` elements
in each dimension :math:`i` are represented as :math:`1`-dimensional arrays with
:math:`\prod_{i=1}^N n_i` elements.
"""


from ._freud import Index2D
from ._freud import Index3D
