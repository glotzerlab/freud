import numpy as np
import freud
import util
from benchmark import Benchmark
from benchmarker import run_benchmarks


class BenchmarkEnvironmentBondOrder(Benchmark):
    def __init__(self, num_neighbors, bins):
        self.num_neighbors = num_neighbors
        self.bins = bins

    def bench_setup(self, N):
        n = N
        np.random.seed(0)
        (self.box, self.positions) = util.make_fcc(n, n, n)
        self.random_quats = np.random.rand(len(self.positions), 4)
        self.random_quats /= np.linalg.norm(self.random_quats,
                                            axis=1)[:, np.newaxis]

        self.bo = freud.environment.BondOrder(self.bins)

    def bench_run(self, N):
        self.bo.compute((self.box, self.positions), self.random_quats,
                        neighbors={'num_neighbors': self.num_neighbors})


def run():
    Ns = [4, 8, 16]
    num_neighbors = 12
    bins = (6, 6)
    number = 100

    name = 'freud.environment.BondOrder'
    classobj = BenchmarkEnvironmentBondOrder
    return run_benchmarks(name, Ns, number, classobj,
                          num_neighbors=num_neighbors,
                          bins=bins)


if __name__ == '__main__':
    run()
