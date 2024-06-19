# Copyright (c) 2010-2024 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np
import rowan
from benchmark import Benchmark
from benchmarker import run_benchmarks

import freud


class BenchmarkOrderRotationalAutocorrelation(Benchmark):
    def __init__(self, sph_l):
        self.sph_l = sph_l

    def bench_setup(self, N):
        seed = 0
        np.random.seed(seed)
        self.orientations = rowan.random.random_sample((N,))
        self.ra = freud.order.RotationalAutocorrelation(self.sph_l)

    def bench_run(self, N):
        self.ra.compute(self.orientations, self.orientations)


def run():
    Ns = [1000, 5000, 10000]
    number = 100
    name = "freud.order.RotationalAutocorrelation"

    kwargs = {"sph_l": 2}

    return run_benchmarks(
        name, Ns, number, BenchmarkOrderRotationalAutocorrelation, **kwargs
    )


if __name__ == "__main__":
    run()
