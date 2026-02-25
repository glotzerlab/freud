# Copyright (c) 2010-2026 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np
from benchmark import Benchmark
from benchmarker import run_benchmarks

import freud


class BenchmarkPeriodicBuffer(Benchmark):
    def __init__(self, L, buf, images):
        self.L = L
        self.buffer = buf
        self.images = images

    def bench_setup(self, N):
        seed = 0
        np.random.seed(seed)
        self.positions = np.random.uniform(-self.L / 2, self.L / 2, (N, 3))
        self.pbuff = freud.locality.PeriodicBuffer()

    def bench_run(self, N):
        box = freud.box.Box.cube(self.L)
        self.pbuff.compute(
            (box, self.positions), buffer=self.buffer, images=self.images
        )


def run():
    Ns = [1000, 5000, 10000]
    number = 100
    name = "freud.locality.PeriodicBuffer"

    L = 10
    buf = 2
    images = True
    return run_benchmarks(
        name, Ns, number, BenchmarkPeriodicBuffer, L=L, buf=buf, images=images
    )


if __name__ == "__main__":
    run()
