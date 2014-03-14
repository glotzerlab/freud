import numpy
import multiprocessing
from _freud import setNumThreads
from _freud import Lind

class lindex(object):
    """docstring for lindex"""
    def __init__(self, box, rmax, dr, nthreads=None):
        super(lindex, self).__init__()
        self.box = box
        self.rmax = rmax
        self.dr = dr
        self.lind_handle = Lind(self.box, self.rmax, self.dr)
        if nthreads is not None:
            setNumThreads(int(nthreads))
        else:
            setNumThreads(multiprocessing.cpu_count())

    def compute(self, pos):
        self.pos = pos
        self.lind_handle.compute(self.pos)
        self.lind_array = numpy.copy(self.lind_handle.getLindexArray())
        self.lindex = numpy.mean(self.lind_array)
