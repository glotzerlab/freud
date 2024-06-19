# Copyright (c) 2010-2024 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np
from benchmark import Benchmark
from benchmarker import run_benchmarks

import freud


class BenchmarkDensityGaussianDensity(Benchmark):
    def __init__(self, width, r_max, sigma):
        self.width = width
        self.r_max = r_max
        self.sigma = sigma

    def bench_setup(self, N):
        self.box_size = self.r_max * 20
        self.box = freud.box.Box.square(self.box_size)
        np.random.seed(0)
        self.points = (
            np.random.random_sample((N, 3)).astype(np.float32) * self.box_size
            - self.box_size / 2
        )
        self.points[:, 2] = 0
        self.gd = freud.density.GaussianDensity(self.width, self.r_max, self.sigma)

    def bench_run(self, N):
        self.gd.compute((self.box, self.points))


def run():
    Ns = [1000, 10000]
    width = 100
    r_max = 1
    sigma = 0.1
    name = "freud.density.GaussianDensity"
    classobj = BenchmarkDensityGaussianDensity
    number = 100

    return run_benchmarks(
        name, Ns, number, classobj, width=width, r_max=r_max, sigma=sigma
    )


if __name__ == "__main__":
    run()
