# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

## \package freud.locality
#
# Methods and data structures computing properties that at local in space.
#

# bring related C++ classes into the locality module
from ._freud import LinkCell
from ._freud import IteratorLinkCell
from ._freud import NearestNeighbors
from ._freud import NeighborList
