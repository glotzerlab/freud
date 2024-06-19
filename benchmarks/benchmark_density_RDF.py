# Copyright (c) 2010-2024 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np
from benchmark import Benchmark
from benchmarker import run_benchmarks

import freud


class BenchmarkDensityRDF(Benchmark):
    def __init__(self, r_max, bins, r_min):
        self.r_max = r_max
        self.bins = bins
        self.r_min = r_min

    def bench_setup(self, N):
        self.box_size = self.r_max * 3.1
        np.random.seed(0)
        self.points = (
            np.random.random_sample((N, 3)).astype(np.float32) * self.box_size
            - self.box_size / 2
        )
        self.rdf = freud.density.RDF(self.bins, self.r_max, r_min=self.r_min)
        self.box = freud.box.Box.cube(self.box_size)

    def bench_run(self, N):
        self.rdf.compute((self.box, self.points), reset=False)
        self.rdf.compute((self.box, self.points))


def run():
    Ns = [1000, 10000]
    r_max = 10.0
    bins = 10
    r_min = 0
    number = 100
    name = "freud.density.RDF"
    classobj = BenchmarkDensityRDF

    return run_benchmarks(
        name, Ns, number, classobj, r_max=r_max, bins=bins, r_min=r_min
    )


if __name__ == "__main__":
    run()
