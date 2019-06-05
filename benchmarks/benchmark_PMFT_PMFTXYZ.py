import numpy as np
import freud
from benchmark import Benchmark
from benchmarker import run_benchmarks
import rowan


class BenchmarkPMFTPMFTXYZ(Benchmark):
    def __init__(self, L, x_max, y_max, z_max, n_x, n_y, n_z):
        self.L = L
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z

    def bench_setup(self, N):
        self.box = freud.box.Box.cube(self.L)
        seed = 0
        np.random.seed(seed)
        self.points = np.random.uniform(-self.L/2, self.L/2, (N, 3))
        self.orientations = rowan.random.random_sample((N, ))
        self.pmft = freud.pmft.PMFTXYZ(self.x_max, self.y_max, self.z_max,
                                       self.n_x, self.n_y, self.n_z)

    def bench_run(self, N):
        self.pmft.compute(self.box, self.points, self.orientations)
        self.pmft.bin_counts


def run():
    Ns = [100, 500, 1000, 2000]
    number = 100
    name = 'freud.PMFT.PMFTXYZ'

    kwargs = {"L": 25.0,
              "x_max": 5.23,
              "y_max": 6.23,
              "z_max": 7.23,
              "n_x": 100,
              "n_y": 110,
              "n_z": 120}

    return run_benchmarks(name, Ns, number, BenchmarkPMFTPMFTXYZ,
                          **kwargs)


if __name__ == '__main__':
    run()
