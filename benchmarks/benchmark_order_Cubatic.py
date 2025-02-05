# Copyright (c) 2010-2025 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np
import rowan
from benchmark import Benchmark
from benchmarker import run_benchmarks

import freud


class BenchmarkOrderCubatic(Benchmark):
    def __init__(self, t_initial, t_final, scale, n_replicates, seed):
        self.t_initial = t_initial
        self.t_final = t_final
        self.scale = scale
        self.n_replicates = n_replicates
        self.seed = seed

    def bench_setup(self, N):
        np.random.seed(0)
        self.orientations = rowan.random.random_sample((N,))
        self.cop = freud.order.Cubatic(
            self.t_initial, self.t_final, self.scale, self.n_replicates, self.seed
        )

    def bench_run(self, N):
        self.cop.compute(self.orientations)


def run():
    Ns = [1000, 5000, 10000]
    number = 100
    name = "freud.order.Cubatic"

    kwargs = {
        "t_initial": 5.0,
        "t_final": 0.001,
        "scale": 0.95,
        "n_replicates": 10,
        "seed": 0,
    }

    return run_benchmarks(name, Ns, number, BenchmarkOrderCubatic, **kwargs)


if __name__ == "__main__":
    run()
