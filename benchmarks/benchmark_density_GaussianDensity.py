import numpy as np
import numpy.testing as npt
import freud
import unittest
from benchmark import Benchmark


class BenchmarkDensityGaussianDensity(Benchmark):
    def __init__(self, width, rcut, sigma):
        self.width = width
        self.rcut = rcut
        self.sigma = sigma

    def bench_setup(self, N):
        self.box_size = self.rcut*3.1
        np.random.seed(0)
        self.points = np.random.random_sample((N, 3)).astype(np.float32) \
            * self.box_size - self.box_size/2
        self.points[:, 2] = 0
        self.diff = freud.density.GaussianDensity(self.width, self.rcut,
                                                  self.sigma)

    def bench_run(self, N):
        testBox = freud.box.Box.square(self.box_size)
        self.diff.compute(testBox, self.points)
