import numpy as np
from benchmark import Benchmark
from benchmarker import run_benchmarks

import freud


class BenchmarkOrderTranslational(Benchmark):
    def __init__(self, L, r_max, k):
        self.L = L
        self.r_max = r_max
        self.k = k

    def bench_setup(self, N):
        self.box = freud.box.Box.square(self.L)
        seed = 0
        np.random.seed(seed)
        self.points = np.asarray(
            np.random.uniform(-self.L / 2, self.L / 2, (N, 3)), dtype=np.float32
        )
        self.points[:, 2] = 0.0
        self.top = freud.order.Translational(self.k)

    def bench_run(self, N):
        self.top.compute((self.box, self.points), {"r_max": self.r_max})


def run():
    Ns = [100, 500, 1000, 5000, 10000]
    number = 100
    name = "freud.order.Translational"

    kwargs = {"L": 10, "r_max": 3, "k": 6}

    return run_benchmarks(name, Ns, number, BenchmarkOrderTranslational, **kwargs)


if __name__ == "__main__":
    run()
