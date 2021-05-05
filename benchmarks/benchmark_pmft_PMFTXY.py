import numpy as np
from benchmark import Benchmark
from benchmarker import run_benchmarks

import freud


class BenchmarkPMFTPMFTXY(Benchmark):
    def __init__(self, L, x_max, y_max, bins):
        self.L = L
        self.x_max = x_max
        self.y_max = y_max
        self.bins = bins

    def bench_setup(self, N):
        self.box = freud.box.Box.square(self.L)
        seed = 0
        np.random.seed(seed)
        self.points = np.random.uniform(-self.L / 2, self.L / 2, (N, 3))
        self.points[:, 2] = 0
        self.orientations = np.random.uniform(0.0, 2 * np.pi, (N, 1))
        self.pmft = freud.pmft.PMFTXY(self.x_max, self.y_max, self.bins)

    def bench_run(self, N):
        self.pmft.compute((self.box, self.points), self.orientations)
        self.pmft.bin_counts


def run():
    Ns = [100, 500, 1000, 2000]
    number = 100
    name = "freud.PMFT.PMFTXY"

    kwargs = {
        "L": 16.0,
        "x_max": 3.6,
        "y_max": 4.2,
        "bins": (100, 110),
    }

    return run_benchmarks(name, Ns, number, BenchmarkPMFTPMFTXY, **kwargs)


if __name__ == "__main__":
    run()
