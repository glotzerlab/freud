# Copyright (c) 2010-2026 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np
from benchmark import Benchmark
from benchmarker import run_benchmarks

import freud


class BenchmarkOrderSolidLiquid(Benchmark):
    def __init__(self, L, r_max, Qthreshold, Sthreshold, sph_l):
        self.L = L
        self.r_max = r_max
        self.Qthreshold = Qthreshold
        self.Sthreshold = Sthreshold
        self.sph_l = sph_l

    def bench_setup(self, N):
        self.box = freud.box.Box.cube(self.L)
        seed = 0
        np.random.seed(seed)
        self.points = np.asarray(
            np.random.uniform(-self.L / 2, self.L / 2, (N, 3)), dtype=np.float32
        )
        self.sl = freud.order.SolidLiquid(self.sph_l, self.Qthreshold, self.Sthreshold)

    def bench_run(self, N):
        self.sl.compute((self.box, self.points), neighbors={"r_max": self.r_max})


def run():
    Ns = [100, 500, 1000, 5000]
    number = 100
    name = "freud.order.SolidLiquid"

    kwargs = {"L": 10, "r_max": 2, "Qthreshold": 0.7, "Sthreshold": 6, "sph_l": 6}

    return run_benchmarks(name, Ns, number, BenchmarkOrderSolidLiquid, **kwargs)


if __name__ == "__main__":
    run()
