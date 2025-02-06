# Copyright (c) 2010-2025 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np
from benchmark import Benchmark
from benchmarker import run_benchmarks

import freud


class BenchmarkDensityCorrelationFunction(Benchmark):
    def __init__(self, bins, rmax):
        self.rmax = rmax
        self.bins = bins

    def bench_setup(self, N):
        self.box_size = self.rmax * 3.1
        np.random.seed(0)
        self.points = (
            np.random.random_sample((N, 3)).astype(np.float32) * self.box_size
            - self.box_size / 2
        )
        self.points[:, 2] = 0
        ang = np.random.random_sample(N).astype(np.float64) * 2.0 * np.pi
        self.comp = np.exp(1j * ang)
        self.ocf = freud.density.CorrelationFunction(self.bins, self.bins)
        self.box = freud.box.Box.square(self.box_size)

    def bench_run(self, N):
        self.ocf.compute(
            (self.box, self.points),
            self.comp,
            self.points,
            np.conj(self.comp),
            reset=False,
        )
        self.ocf.compute(
            (self.box, self.points), self.comp, self.points, np.conj(self.comp)
        )


def run():
    Ns = [1000, 10000]
    rmax = 10.0
    bins = 10
    name = "freud.density.CorrelationFunction"
    classobj = BenchmarkDensityCorrelationFunction
    number = 100

    return run_benchmarks(name, Ns, number, classobj, rmax=rmax, bins=bins)


if __name__ == "__main__":
    run()
