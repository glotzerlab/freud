import numpy as np
import freud
from benchmark import Benchmark
from benchmarker import run_benchmarks


class BenchmarkOrderTransOrderParameter(Benchmark):
    def __init__(self, L, rmax):
        self.L = L
        self.rmax = rmax

    def bench_setup(self, N):
        self.box = freud.box.Box.square(self.L)
        seed = 0
        np.random.seed(seed)
        self.points = np.asarray(np.random.uniform(-self.L/2, self.L/2,
                                                   (N, 3)),
                                 dtype=np.float32)
        self.points[:, 2] = 0.0
        self.top = freud.order.TransOrderParameter(self.rmax)

    def bench_run(self, N):
        self.top.compute(self.box, self.points)


def run():
    Ns = [100, 500, 1000, 5000, 10000]
    print_stats = True
    number = 100
    name = 'freud.order.TransOrderParameter'

    kwargs = {"L": 10,
              "rmax": 3}

    return run_benchmarks(name, Ns, number,
                          BenchmarkOrderTransOrderParameter,
                          print_stats, **kwargs)


if __name__ == '__main__':
    run()
