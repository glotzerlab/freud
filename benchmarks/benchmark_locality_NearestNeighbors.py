import numpy as np
import freud
from benchmark import Benchmark
from benchmarker import do_some_benchmarks


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


def run(on_circleci=False):
    Ns = [1000, 10000]
    rcut = 0.5
    L = 10
    num_neighbors = 6
    print_stats = True
    number = 100

    name = 'freud.locality.NearestNeighbors'
    classobj = BenchmarkLocalityNearestNeighbors
    return do_some_benchmarks(name, Ns, number, classobj, print_stats,
                              on_circleci,
                              L=L, rcut=rcut, num_neighbors=num_neighbors)
