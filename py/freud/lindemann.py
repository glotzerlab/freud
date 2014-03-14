import numpy

from _freud import setNumThreads
from _freud import lindemann

class lindex:
    """docstring for lindex"""
    def __init__(self, box, rmax, dr):
        super(lindex, self).__init__()
        self.box = box
        self.rmax = rmax
        self.dr = dr
        self.lind_handle = lindemann.Lind(self.box, self.rmax, self.dr)

    def compute(self, pos):
        self.pos = pos
        self.lind_handle.compute(self.pos))
        self.lind_array = numpy.copy(self.lind_handle.getLindexArray())
        self.lindex = numpy.mean(self.lind_array)
