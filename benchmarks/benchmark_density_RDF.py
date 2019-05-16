import numpy as np
import numpy.testing as npt
import freud
import unittest
from benchmark import Benchmark


class BenchmarkDensityRDF(Benchmark):
    def __init__(self, rmax, dr, rmin):
        self.rmax = rmax
        self.dr = dr
        self.rmin = rmin

    def bench_setup(self, N):
        self.box_size = self.rmax*3.1
        np.random.seed(0)
        self.points = np.random.random_sample((N, 3)).astype(np.float32) \
            * self.box_size - self.box_size/2
        self.points.flags['WRITEABLE'] = False
        self.rdf = freud.density.RDF(self.rmax, self.dr, rmin=self.rmin)

    def bench_run(self, N):
        self.rdf.accumulate(freud.box.Box.cube(self.box_size), self.points)
        self.rdf.compute(freud.box.Box.cube(self.box_size), self.points)
