import numpy as np
import numpy.testing as npt
import freud
import unittest
from benchmark import Benchmark


class BenchmarkDensityComplexCF(Benchmark):
    def __init__(self, rmax, dr):
        self.rmax = rmax
        self.dr = dr

    def bench_setup(self, N):
        self.box_size = self.rmax*3.1
        np.random.seed(0)
        self.points = np.random.random_sample((N, 3)).astype(np.float32) \
            * self.box_size - self.box_size/2
        ang = np.random.random_sample((N)).astype(np.float64) \
            * 2.0 * np.pi
        self.comp = np.exp(1j*ang)
        self.ocf = freud.density.ComplexCF(self.rmax, self.dr)

    def bench_run(self, N):
        self.ocf.accumulate(freud.box.Box.square(self.box_size), self.points,
                            self.comp, self.points, np.conj(self.comp))
        self.ocf.compute(freud.box.Box.square(self.box_size), self.points,
                         self.comp, self.points, np.conj(self.comp))
