# Copyright (c) 2010-2025 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

# density,
from . import (
    box,
    cluster,
    data,
    density,
    diffraction,
    environment,
    interface,
    locality,
    msd,
    order,
    parallel,
    pmft,
)
from .box import Box
from .locality import AABBQuery, LinkCell, NeighborList
from .parallel import NumThreads, get_num_threads, set_num_threads

# Override TBB's default autoselection. This is necessary because once the
# automatic selection runs, the user cannot change it.
set_num_threads(0)

__version__ = "3.4.0"

__all__ = [
    "AABBQuery",
    "Box",
    "LinkCell",
    "NeighborList",
    "NumThreads",
    "__version__",
    "box",
    "cluster",
    "data",
    "density",
    "diffraction",
    "environment",
    "get_num_threads",
    "interface",
    "locality",
    "msd",
    "order",
    "parallel",
    "pmft",
    "set_num_threads",
]

__citation__ = """@article{freud2020,
    title = {freud: A Software Suite for High Throughput
             Analysis of Particle Simulation Data},
    author = {Vyas Ramasubramani and
              Bradley D. Dice and
              Eric S. Harper and
              Matthew P. Spellings and
              Joshua A. Anderson and
              Sharon C. Glotzer},
    journal = {Computer Physics Communications},
    volume = {254},
    pages = {107275},
    year = {2020},
    issn = {0010-4655},
    doi = {https://doi.org/10.1016/j.cpc.2020.107275},
    url = {http://www.sciencedirect.com/science/article/pii/S0010465520300916},
    keywords = {Simulation analysis, Molecular dynamics, Monte Carlo,
                Computational materials science},
}"""
