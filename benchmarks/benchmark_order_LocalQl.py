import numpy as np
import freud
from benchmark import Benchmark
from benchmarker import run_benchmarks


class BenchmarkOrderLocalQl(Benchmark):
    def __init__(self, L, rmax, I):
        self.L = L
        self.rmax = rmax
        self.I = I # noqa

    def bench_setup(self, N):
        box = freud.box.Box.cube(self.L)
        seed = 0
        np.random.seed(seed)
        self.points = np.asarray(np.random.uniform(-self.L/2, self.L/2,
                                                   (N, 3)),
                                 dtype=np.float32)
        self.lql = freud.order.LocalQl(box, self.rmax, self.I)

    def bench_run(self, N):
        self.lql.compute(self.points)
        self.lql.computeAve(self.points)


def run():
    Ns = [100, 500, 1000, 5000]
    number = 100
    name = 'freud.order.LocalQl'

    kwargs = {"L": 10,
              "rmax": 1.5,
              "I": 6}

    return run_benchmarks(name, Ns, number, BenchmarkOrderLocalQl,
                          **kwargs)


if __name__ == '__main__':
    run()
