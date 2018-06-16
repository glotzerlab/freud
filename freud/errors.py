# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

# \package freud.errors
#
# Errors and exceptions internal to freud

import warnings
class FreudDeprecationWarning(UserWarning):
    """Raised when a freud features is pending deprecation"""
    pass

