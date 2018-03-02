# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

# __init__.py marks this directory as a python module

try:
    from . import _freud
except ImportError as e:
    print("""
******************** WARNING **********************
Could not find a compiled _freud module. You may
have imported freud from the source directory.
freud must be compiled and installed to function.
Set your PYTHONPATH appropriately and change to a
different directory before importing freud.
******************** WARNING **********************
""")

from . import parallel
from . import box
from . import bond
from . import cluster
from . import density
from . import kspace
from . import locality
from . import order
from . import interface
# from . import shape
from . import voronoi
from . import pmft
from . import index
from . import common

__version__ = '0.7.0'
