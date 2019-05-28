import numpy as np
import freud
from benchmark import Benchmark
from benchmarker import run_benchmarks


class BenchmarkLocalityNearestNeighbors(Benchmark):
    def __init__(self, L, rcut, num_neighbors):
        self.L = L
        self.rcut = rcut
        self.num_neighbors = num_neighbors

    def bench_setup(self, N):
        self.fbox = freud.box.Box.cube(self.L)
        self.cl = freud.locality.NearestNeighbors(self.rcut,
                                                  self.num_neighbors)

        seed = 0
        np.random.seed(seed)
        self.points = np.random.uniform(-self.L/2, self.L/2,
                                        (N, 3)).astype(np.float32)

    def bench_run(self, N):
        self.cl.compute(self.fbox, self.points, self.points)


def run():
    Ns = [1000, 10000]
    rcut = 0.5
    L = 10
    num_neighbors = 6
    number = 100

    name = 'freud.locality.NearestNeighbors'
    classobj = BenchmarkLocalityNearestNeighbors
    return run_benchmarks(name, Ns, number, classobj,
                          L=L, rcut=rcut, num_neighbors=num_neighbors)


if __name__ == '__main__':
    run()
