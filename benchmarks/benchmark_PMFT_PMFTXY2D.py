import numpy as np
import freud
from benchmark import Benchmark
from benchmarker import run_benchmarks


class BenchmarkPMFTPMFTXY2D(Benchmark):
    def __init__(self, L, x_max, y_max, n_x, n_y):
        self.L = L
        self.x_max = x_max
        self.y_max = y_max
        self.n_x = n_x
        self.n_y = n_y

    def bench_setup(self, N):
        self.box = freud.box.Box.square(self.L)
        seed = 0
        np.random.seed(seed)
        self.points = np.random.uniform(-self.L/2, self.L/2, (N, 3))
        self.points[:, 2] = 0
        self.orientations = np.random.uniform(0.0, 2*np.pi, (N, 1))
        self.pmft = freud.pmft.PMFTXY2D(self.x_max, self.y_max,
                                        self.n_x, self.n_y)

    def bench_run(self, N):
        self.pmft.compute(self.box, self.points, self.orientations)
        self.pmft.bin_counts


def run():
    Ns = [100, 500, 1000, 2000]
    number = 100
    name = 'freud.PMFT.PMFTXY2D'

    kwargs = {"L": 16.0,
              "x_max": 3.6,
              "y_max": 4.2,
              "n_x": 100,
              "n_y": 110}

    return run_benchmarks(name, Ns, number, BenchmarkPMFTPMFTXY2D,
                          **kwargs)


if __name__ == '__main__':
    run()
