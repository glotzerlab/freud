# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from . import box
from . import cluster
from . import common
from . import density
from . import environment
from . import interface
from . import locality
from . import msd
from . import order
from . import parallel
from . import pmft

# Override TBB's default autoselection. This is necessary because once the
# automatic selection runs, the user cannot change it.
parallel.setNumThreads(0)

__version__ = '1.2.2'

__all__ = [
    '__version__',
    'box',
    'cluster',
    'common',
    'density',
    'environment',
    'interface',
    'locality',
    'msd',
    'order',
    'parallel',
    'pmft',
    'voronoi',
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
