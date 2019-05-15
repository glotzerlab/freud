import numpy as np
from freud import locality, box
from benchmark import Benchmark


class BenchmarkLocalityNearestNeighbors(Benchmark):
    def __init__(self, L, rcut, num_neighbors):
        self.L = L
        self.rcut = rcut
        self.num_neighbors = num_neighbors

    def bench_setup(self, N):
        self.fbox = box.Box.cube(self.L)
        self.cl = locality.NearestNeighbors(self.rcut, self.num_neighbors)

        seed = 0
        np.random.seed(seed)
        self.points = np.random.uniform(-self.L/2, self.L/2,
                                        (N, 3)).astype(np.float32)

    def bench_run(self, N):
        self.cl.compute(self.fbox, self.points, self.points)
