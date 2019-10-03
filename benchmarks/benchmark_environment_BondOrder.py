import numpy as np
import freud
import util
from benchmark import Benchmark
from benchmarker import run_benchmarks


class BenchmarkEnvironmentBondOrder(Benchmark):
    def __init__(self, rmax, k, num_neighbors, n_bins_t, n_bins_p):
        self.rmax = rmax
        self.num_neighbors = num_neighbors
        self.n_bins_t = n_bins_t
        self.n_bins_p = n_bins_p

    def bench_setup(self, N):
        n = N
        np.random.seed(0)
        (self.box, self.positions) = util.make_fcc(n, n, n)
        self.random_quats = np.random.rand(len(self.positions), 4)
        self.random_quats /= np.linalg.norm(self.random_quats,
                                            axis=1)[:, np.newaxis]

        self.bo = freud.environment.BondOrder(self.rmax,
                                              self.num_neighbors,
                                              self.n_bins_t, self.n_bins_p)

    def bench_run(self, N):
        self.bo.compute(self.box, self.positions, self.random_quats,
                        self.positions, self.random_quats)


def run():
    Ns = [4, 8, 16]
    r_cut = 1.5
    num_neighbors = 12
    npt = npp = 6
    number = 100

    name = 'freud.environment.BondOrder'
    classobj = BenchmarkEnvironmentBondOrder
    return run_benchmarks(name, Ns, number, classobj,
                          rmax=r_cut, k=0,
                          num_neighbors=num_neighbors,
                          n_bins_t=npt, n_bins_p=npp)


if __name__ == '__main__':
    run()
