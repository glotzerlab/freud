#!/usr/bin/env python
# convert input stream of polyhedron vertices from Mathematica to freud shape definition file
# argument: name of shape
#
# Mathematice data is obtained with calls like
# ExportString[PolyhedronData["Cube", "VertexCoordinates"], "Table", "FieldSeparators" -> ", "]
# or
# ExportString[PolyhedronData[{"Prism", 6}, "VertexCoordinates"], "Table", "FieldSeparators" -> ", "]
# and the results cut and pasted to terminal input to this script.
# Input is terminated with a newline and end-of-file character, i.e. CTRL-D
#
# Numeric interpretation of Mathematica data may be necessary where translation to Python isn't as easy.
# E.g.
# ExportString[N[PolyhedronData["ObtuseGoldenRhombohedron", "VertexCoordinates"]], "Table", "FieldSeparators" -> ", "]
#

# open the output file for writing
import sys
name = sys.argv[1]
outfile = open(name+'.py', 'w')

# Set up some boiler plate

header = """from __future__ import division
from numpy import sqrt
import numpy
from freud.shape import ConvexPolyhedron

# Example:
"""

example = "# from freud.shape.{0} import shape\n".format(name)

footer = """         ]

shape = ConvexPolyhedron(numpy.array(points))
"""

# Process the input

pstrings = list()
instring = sys.stdin.read()
# Strip out quotes
instring = instring.replace('"', '')
# merge wrapped lines
instring = instring.replace('\\\n', '')
# split input into list of lines
lines = instring.splitlines()
for line in lines:
    # Turn Mathematica syntax into Python syntax
    line = line.replace('Sqrt','sqrt')
    line = line.replace('[','(').replace(']',')')
    line = line.replace('^','**')
    # get string values of x,y,z
    x,y,z = line.split(', ')
    pstring = "          ({x}, {y}, {z}),\n".format(x=x,y=y,z=z)
    pstrings.append(pstring)

# Write the output

outfile.write(header)
outfile.write(example)
outfile.write("points = [ \n")

outfile.writelines(pstrings)

outfile.write(footer)
outfile.close()
