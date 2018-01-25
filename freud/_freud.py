# Copyright (c) 2010-2016 The Regents of the University of Michigan
# This file is part of the Freud project, released under the BSD 3-Clause License.

# This file exists to allow the freud module to import from the source checkout dir
# for use when building the sphinx documentation.
print("""
******************** WARNING *********************
You have imported freud from the source directory.
Freud must be compiled and installed to function.
Set your PYTHONPATH appropriately and change to a
different directory before importing freud.
******************** WARNING *********************
""")
