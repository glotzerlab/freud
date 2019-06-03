import numpy as np
import freud
from benchmark import Benchmark
from benchmarker import run_benchmarks


class BenchmarkPMFTPMFTXYT(Benchmark):
    def __init__(self, L, x_max, y_max, n_x, n_y, n_t):
        self.L = L
        self.x_max = x_max
        self.y_max = y_max
        self.n_x = n_x
        self.n_y = n_y
        self.n_t = n_t

    def bench_setup(self, N):
        self.box = freud.box.Box.square(self.L)
        seed = 0
        np.random.seed(seed)
        self.points = np.random.uniform(-self.L/2, self.L/2, (N, 3))
        self.points[:, 2] = 0
        self.orientations = np.random.uniform(0.0, 2*np.pi, (N, 1))
        self.pmft = freud.pmft.PMFTXYT(self.x_max, self.y_max,
                                       self.n_x, self.n_y, self.n_t)

    def bench_run(self, N):
        self.pmft.compute(self.box, self.points, self.orientations)


def run():
    Ns = [100, 500, 1000, 2000]
    number = 100
    name = 'freud.PMFT.PMFTXYT'

    kwargs = {"L": 16.0,
              "x_max": 3.6,
              "y_max": 4.2,
              "n_x": 20,
              "n_y": 30,
              "n_t": 40}

    return run_benchmarks(name, Ns, number, BenchmarkPMFTPMFTXYT,
                          **kwargs)


if __name__ == '__main__':
    run()
