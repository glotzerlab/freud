import numpy as np
from freud import locality, box
from benchmark import Benchmark


class BenchmarkLocalityAABBQuery(Benchmark):
    def __init__(self, L, rcut):
        self.L = L
        self.rcut = rcut

    def bench_setup(self, N):
        self.fbox = box.Box.cube(self.L)
        seed = 0
        np.random.seed(seed)
        self.points = np.random.uniform(-self.L/2, self.L/2, (N, 3))

    def bench_run(self, N):
        self.aq = locality.AABBQuery(self.fbox, self.points)
        self.aq.queryBall(self.points, self.rcut, exclude_ii=True)
