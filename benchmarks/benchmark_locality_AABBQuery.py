import numpy as np
import freud
from benchmark import Benchmark
from benchmarker import run_benchmarks


class BenchmarkLocalityAABBQuery(Benchmark):
    def __init__(self, L, rcut):
        self.L = L
        self.rcut = rcut

    def bench_setup(self, N):
        self.box = freud.box.Box.cube(self.L)
        seed = 0
        np.random.seed(seed)
        self.points = np.random.uniform(-self.L/2, self.L/2, (N, 3))

    def bench_run(self, N):
        aq = freud.locality.AABBQuery(self.box, self.points)
        aq.queryBall(self.points, self.rcut, exclude_ii=True)


def run():
    Ns = [1000, 10000]
    rcut = 0.5
    L = 10
    number = 100

    name = 'freud.locality.AABBQuery'
    return run_benchmarks(name, Ns, number, BenchmarkLocalityAABBQuery,
                          L=L, rcut=rcut)


if __name__ == '__main__':
    run()
