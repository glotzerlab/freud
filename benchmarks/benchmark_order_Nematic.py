# Copyright (c) 2010-2025 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np
import rowan
from benchmark import Benchmark
from benchmarker import run_benchmarks

import freud


class BenchmarkOrderNematic(Benchmark):
    def __init__(self, u):
        self.u = u

    def bench_setup(self, N):
        seed = 0
        np.random.seed(seed)
        self.orientations = rowan.random.random_sample((N,))
        self.nop = freud.order.Nematic(np.array(self.u))

    def bench_run(self, N):
        self.nop.compute(self.orientations)


def run():
    Ns = [1000, 5000, 10000]
    number = 100
    name = "freud.order.Nematic"

    kwargs = {"u": [1, 0, 0]}

    return run_benchmarks(name, Ns, number, BenchmarkOrderNematic, **kwargs)


if __name__ == "__main__":
    run()
