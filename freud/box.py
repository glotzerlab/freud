# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The box module provides the Box class, which defines the geometry of the
simulation box. The module natively supports periodicity by providing the
fundamental features for wrapping vectors outside the box back into it.
"""

from ._freud import Box
