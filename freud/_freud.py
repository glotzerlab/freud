# Copyright (c) 2010-2016 The Regents of the University of Michigan
# This file is part of the Freud project, released under the BSD 3-Clause License.

# This file exists to allow the freud module to import from the source checkout dir
# for use when building the sphinx documentation.
print()
print("******************** WARNING ********************")
print("You have imported freud from the source directory.")
print("Freud must be compiled and installed to function.")
print("Set your PYTHONPATH appropriately and cd to a ")
print("a different directory before importing freud. ")
print("******************** WARNING ********************")
print()
