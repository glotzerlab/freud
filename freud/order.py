# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

R"""
The order module contains functions which compute order parameters for the
whole system or individual particles. Order parameters take bond order data and
interpret it in some way to quantify the degree of order in a system using a
scalar value. This is often done through computing spherical harmonics of the
bond order diagram, which are the spherical analogue of Fourier Transforms.
"""

import warnings
from .errors import FreudDeprecationWarning

from ._freud import CubaticOrderParameter
from ._freud import NematicOrderParameter
from ._freud import HexOrderParameter
from ._freud import TransOrderParameter

# everything below uses spherical harmonics
from ._freud import LocalQl
from ._freud import LocalQlNear
from ._freud import LocalWl
from ._freud import LocalWlNear
from ._freud import SolLiq
from ._freud import SolLiqNear

# The below are maintained for backwards compatibility
# but have been moved to the environment module
from ._freud import BondOrder as _EBO
from ._freud import LocalDescriptors as _ELD
from ._freud import MatchEnv as _EME
from ._freud import Pairing2D as _EP
from ._freud import AngularSeparation as _EAS


class BondOrder(_EBO):
    """
    .. note::
        This class is only retained for backwards compatibility.
        Please use :py:class:`freud.environment.BondOrder` instead.

    .. deprecated:: 0.8.2
       Use :py:class:`freud.environment.BondOrder` instead.

    """
    def __init__(self, rmax, k, n, n_bins_t, n_bins_p):
        warnings.warn("This class is deprecated, use "
                      "freud.environment.BondOrder instead!",
                      FreudDeprecationWarning)


class LocalDescriptors(_ELD):
    """
    .. note::
        This class is only retained for backwards compatibility.
        Please use :py:class:`freud.environment.LocalDescriptors` instead.

    .. deprecated:: 0.8.2
       Use :py:class:`freud.environment.LocalDescriptors` instead.

    """
    def __init__(self, num_neighbors, lmax, rmax, negative_m=True):
        warnings.warn("This class is deprecated, use "
                      "freud.environment.LocalDescriptors instead!",
                      FreudDeprecationWarning)


class MatchEnv(_EME):
    """
    .. note::
        This class is only retained for backwards compatibility.
        Please use :py:class:`freud.environment.MatchEnv` instead.

    .. deprecated:: 0.8.2
       Use :py:class:`freud.environment.MatchEnv` instead.

    """
    def __init__(self, box, rmax, k):
        warnings.warn("This class is deprecated, use "
                      "freud.environment.MatchEnv instead!",
                      FreudDeprecationWarning)


class Pairing2D(_EP):
    """
    .. note::
        This class is only retained for backwards compatibility.
        Please use :py:mod:`freud.bond` instead.

    .. deprecated:: 0.8.2
       Use :py:mod:`freud.bond` instead.

    """
    def __init__(self, rmax, k, compDotTol):
        warnings.warn("This class is deprecated, use "
                      "freud.bond instead!", FreudDeprecationWarning)


class AngularSeparation(_EAS):
    """
    .. note::
        This class is only retained for backwards compatibility.
        Please use :py:class:`freud.environment.AngularSeparation` instead.

    .. deprecated:: 0.8.2
       Use :py:class:`freud.environment.AngularSeparation` instead.

    """
    def __init__(self, rmax, n):
        warnings.warn("This class is deprecated, use "
                      "freud.environment.AngularSeparation instead!",
                      FreudDeprecationWarning)
