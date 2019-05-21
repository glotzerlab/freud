import numpy as np
import freud
from benchmark import Benchmark
from benchmarker import run_benchmarks


class BenchmarkPMFTPMFTR12(Benchmark):
    def __init__(self, L, r_max, n_r, n_t1, n_t2):
        self.L = L
        self.r_max = r_max
        self.n_r = n_r
        self.n_t1 = n_t1
        self.n_t2 = n_t2

    def bench_setup(self, N):
        self.box = freud.box.Box.square(self.L)
        seed = 0
        np.random.seed(seed)
        self.points = np.random.uniform(-self.L/2, self.L/2, (N, 3))
        self.points[:, 2] = 0
        self.orientations = np.random.uniform(0.0, 2*np.pi, (N, 1))
        self.pmft = freud.pmft.PMFTR12(self.r_max, self.n_r,
                                       self.n_t1, self.n_t2)

    def bench_run(self, N):
        self.pmft.compute(self.box, self.points, self.orientations)


def run():
    Ns = [100, 500, 1000, 2000]
    print_stats = True
    number = 100
    name = 'freud.PMFT.PMFTR12'

    kwargs = {"L": 16.0,
              "r_max": 5.23,
              "n_r": 10,
              "n_t1": 20,
              "n_t2": 30}

    return run_benchmarks(name, Ns, number, BenchmarkPMFTPMFTR12,
                          print_stats, **kwargs)


if __name__ == '__main__':
    run()
