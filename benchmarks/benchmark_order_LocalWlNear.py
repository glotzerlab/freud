import numpy as np
import freud
from benchmark import Benchmark
from benchmarker import run_benchmarks


class BenchmarkOrderLocalWlNear(Benchmark):
    def __init__(self, L, rmax, I, kn):
        self.L = L
        self.rmax = rmax
        self.I = I # noqa
        self.kn = kn

    def bench_setup(self, N):
        box = freud.box.Box.cube(self.L)
        seed = 0
        np.random.seed(seed)
        self.points = np.asarray(np.random.uniform(-self.L/2, self.L/2,
                                                   (N, 3)),
                                 dtype=np.float32)
        self.lwl = freud.order.LocalWlNear(box, self.rmax, self.I, self.kn)

    def bench_run(self, N):
        self.lwl.compute(self.points)
        self.lwl.computeAve(self.points)


def run():
    Ns = [100, 500, 1000, 5000]
    number = 100
    name = 'freud.order.LocalWlNear'

    kwargs = {"L": 10,
              "rmax": 1.5,
              "I": 6,
              "kn": 12}

    return run_benchmarks(name, Ns, number, BenchmarkOrderLocalWlNear,
                          **kwargs)


if __name__ == '__main__':
    run()
