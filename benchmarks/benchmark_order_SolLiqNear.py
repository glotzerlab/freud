import numpy as np
import freud
from benchmark import Benchmark
from benchmarker import run_benchmarks


class BenchmarkOrderSolLiqNear(Benchmark):
    def __init__(self, L, rmax, Qthreshold, Sthreshold, I, kn):
        self.L = L
        self.rmax = rmax
        self.Qthreshold = Qthreshold
        self.Sthreshold = Sthreshold
        self.I = I # noqa
        self.kn = kn

    def bench_setup(self, N):
        box = freud.box.Box.cube(self.L)
        seed = 0
        np.random.seed(seed)
        self.points = np.asarray(np.random.uniform(-self.L/2, self.L/2,
                                                   (N, 3)),
                                 dtype=np.float32)
        self.sl = freud.order.SolLiq(box, self.rmax, self.Qthreshold,
                                     self.Sthreshold, self.I, self.kn)

    def bench_run(self, N):
        self.sl.compute(self.points)


def run():
    Ns = [100, 500, 1000, 5000]
    number = 100
    name = 'freud.order.SolLiqNear'

    kwargs = {"L": 10,
              "rmax": 2,
              "Qthreshold": 0.7,
              "Sthreshold": 6,
              "I": 6,
              "kn": 12}

    return run_benchmarks(name, Ns, number, BenchmarkOrderSolLiqNear,
                          **kwargs)


if __name__ == '__main__':
    run()
