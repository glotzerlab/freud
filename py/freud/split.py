import numpy
import re
from _freud import ShapeSplit

## \package freud.split
#
# Methods to split points

## Create a ShapeSplit object from a list of global points and local points
#
class Split:
    ## Initialize Split:
    # \param box The simulation box
    def __init__(self):
        super(Split, self).__init__()
        self.splitHandle = ShapeSplit()

    ## Split points and reshape into expected form
    # \params positions The positions to be split
    # \params splitPositions The local positions to split points
    def compute(self,
                box,
                positions,
                orientations,
                splitPositions):
        self.splitHandle.compute(box, positions, orientations, splitPositions)
        self.shapePositions = self.splitHandle.getShapeSplit()
        self.shapeOrientations = self.splitHandle.getShapeOrientations()
        self.shapePositions = numpy.copy(self.shapePositions.reshape(len(positions)*len(splitPositions), 3))
        self.shapeOrientations = numpy.copy(self.shapeOrientations.reshape(len(positions)*len(splitPositions), 4))
