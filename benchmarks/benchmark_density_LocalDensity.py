# Copyright (c) 2010-2026 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import math

import numpy as np
from benchmark import Benchmark
from benchmarker import run_benchmarks

import freud


class BenchmarkDensityLocalDensity(Benchmark):
    def __init__(self, nu, rcut):
        self.nu = nu
        self.rcut = rcut

    def bench_setup(self, N):
        box_size = math.sqrt(N * self.nu)
        seed = 0
        np.random.seed(seed)
        self.pos = (
            np.random.random_sample((N, 3)).astype(np.float32) * box_size - box_size / 2
        )
        self.pos[:, 2] = 0
        self.ld = freud.density.LocalDensity(self.rcut, 1)
        box_size = math.sqrt(N * self.nu)
        self.box = freud.box.Box.square(box_size)

    def bench_run(self, N):
        self.ld.compute((self.box, self.pos))


def run():
    Ns = [1000, 10000]
    rcut = 10
    nu = 1
    name = "freud.density.LocalDensity"
    classobj = BenchmarkDensityLocalDensity
    number = 100

    return run_benchmarks(name, Ns, number, classobj, nu=nu, rcut=rcut)


if __name__ == "__main__":
    run()
