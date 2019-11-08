# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from . import box
from . import cluster
from . import data
from . import density
from . import environment
from . import interface
from . import locality
from . import msd
from . import order
from . import parallel
from . import pmft

from .box import Box
from .locality import AABBQuery, LinkCell, NeighborList
from .parallel import get_num_threads, set_num_threads, NumThreads

# Override TBB's default autoselection. This is necessary because once the
# automatic selection runs, the user cannot change it.
set_num_threads(0)

__version__ = '2.0.1'

__all__ = [
    '__version__',
    'box',
    'cluster',
    'data',
    'density',
    'environment',
    'interface',
    'locality',
    'msd',
    'order',
    'parallel',
    'pmft',
    'voronoi',
    'Box',
    'AABBQuery',
    'LinkCell',
    'NeighborList',
    'get_num_threads',
    'set_num_threads',
    'NumThreads',
]

__citation__ = """@misc{freud,
    author = {Vyas Ramasubramani and
              Bradley D. Dice and
              Eric S. Harper and
              Matthew P. Spellings and
              Joshua A. Anderson and
              Sharon C. Glotzer},
    title = {freud: A Software Suite for High Throughput
             Analysis of Particle Simulation Data},
    year = {2019},
    eprint = {arXiv:1906.06317},
}"""
