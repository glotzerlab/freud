from __future__ import print_function
from __future__ import division

import freud
from benchmark import Benchmark
import numpy as np
import math
from benchmarker import do_some_benchmarks


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
        self.ld = freud.density.LocalDensity(self.rcut, 1, 1)

    def bench_run(self, N):
        box_size = math.sqrt(N*self.nu)
        self.ld.compute(freud.box.Box.square(box_size), self.pos)


def run(on_circleci=False):
    Ns = [1000, 10000]
    rcut = 10
    nu = 1
    name = 'freud.density.LocalDensity'
    classobj = BenchmarkDensityLocalDensity
    print_stats = True
    number = 100

    return do_some_benchmarks(name, Ns, number, classobj, print_stats,
                              on_circleci, nu=nu, rcut=rcut)
