# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

R"""
The PMFT Module allows for the calculation of the Potential of Mean Force and Torque (PMFT) [Cit2]_ [Cit3]_ in a number of different coordinate systems.
The PMFT is defined as the negative algorithm of positional correlation function (PCF).
A given set of reference points is given around which the PCF is computed and averaged in a sea of data points.
The resulting values are accumulated in a PCF array listing the value of the PCF at a discrete set of points.
The specific points are determined by the particular coordinate system used to represent the system.

.. note::
    The coordinate system in which the calculation is performed is not the same as the coordinate system in which particle positions and orientations should be supplied. Only certain coordinate systems are available for certain particle positions and orientations:

    * 2D particle coordinates (position: [x, y, 0], orientation: :math:`\theta`):
        * :math:`X`, :math:`Y`.
        * :math:`X`, :math:`Y`, :math:`\theta_2`.
        * :math:`R`, :math:`\theta_1`, :math:`\theta_2`.

    * 3D particle coordinates: :math:`X`, :math:`Y`, :math:`Z`.
"""

from ._freud import PMFTR12
from ._freud import PMFTXYT
from ._freud import PMFTXY2D
from ._freud import PMFTXYZ
