from __future__ import print_function
from __future__ import division

from freud import density, box
from benchmark import Benchmark
import numpy as np
import math


class BenchmarkDensityLocalDensity(Benchmark):
    def __init__(self, nu, rcut):
        self.nu = nu
        self.rcut = rcut

    def bench_setup(self, N):
        box_size = math.sqrt(N*self.nu)
        seed = 0
        np.random.seed(seed)
        self.pos = np.random.random_sample((N, 3)).astype(np.float32) \
            * box_size - box_size/2
        self.pos[:, 2] = 0
        self.ld = density.LocalDensity(self.rcut, 1, 1)

    def bench_run(self, N):
        box_size = math.sqrt(N*self.nu)
        self.ld.compute(box.Box.square(box_size), self.pos)
