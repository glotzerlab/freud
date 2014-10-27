import numpy

from freud.shape import Polygon

class Outline(object):
    def __init__(self, polygon, width):
        """Initialize an outline of a given Polygon object. Takes the
        polygon in question and the outline width to inset."""
        raise RuntimeError("The Outline object has been integrated into the Polygon class. This will be removed in the future.")
