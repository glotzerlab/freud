import numpy as np
import freud
from benchmark import Benchmark
from benchmarker import run_benchmarks


class BenchmarkOrderLocalQlNear(Benchmark):
    def __init__(self, L, rmax, sph_l, kn):
        self.L = L
        self.rmax = rmax
        self.sph_l = sph_l
        self.kn = kn

    def bench_setup(self, N):
        box = freud.box.Box.cube(self.L)
        seed = 0
        np.random.seed(seed)
        self.points = np.asarray(np.random.uniform(-self.L/2, self.L/2,
                                                   (N, 3)),
                                 dtype=np.float32)
        self.lql = freud.order.LocalQlNear(box, self.rmax, self.sph_l, self.kn)

    def bench_run(self, N):
        self.lql.compute(self.points)
        self.lql.computeAve(self.points)


def run():
    Ns = [100, 500, 1000, 5000]
    number = 100
    name = 'freud.order.LocalQlNear'

    kwargs = {"L": 10,
              "rmax": 1.5,
              "sph_l": 6,
              "kn": 12}

    return run_benchmarks(name, Ns, number, BenchmarkOrderLocalQlNear,
                          **kwargs)


if __name__ == '__main__':
    run()
