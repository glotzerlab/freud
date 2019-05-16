import numpy as np
import numpy.testing as npt
import freud
import unittest
from benchmark import Benchmark


class BenchmarkDensityFloatCF(Benchmark):
    def __init__(self, rmax, dr):
        self.rmax = rmax
        self.dr = dr

    def bench_setup(self, N):
        self.box_size = self.rmax*3.1
        np.random.seed(0)
        self.points = np.random.random_sample((N, 3)).astype(np.float32) \
            * self.box_size - self.box_size/2
        self.ang = np.random.random_sample((N)).astype(np.float64) - 0.5
        self.ocf = freud.density.FloatCF(self.rmax, self.dr)

    def bench_run(self, N):
        self.ocf.accumulate(freud.box.Box.square(self.box_size),
                            self.points, self.ang)
        self.ocf.compute(freud.box.Box.square(self.box_size),
                         self.points, self.ang)
