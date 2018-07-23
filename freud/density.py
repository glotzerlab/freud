# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

# \package freud.density
#
# Methods to compute densities from point distributions.
#

R"""
The density module contains various classes relating to the density of the
system. These functions allow evaluation of particle distributions with respect
to other particles.
"""

from ._freud import GaussianDensity
from ._freud import LocalDensity
from ._freud import RDF
from ._freud import ComplexCF
from ._freud import FloatCF
