import numpy as np
import freud
import unittest
import warnings
import util
from benchmark import Benchmark


class BenchmarkEnvironmentBondOrder(Benchmark):
    def __init__(self, rmax, k, num_neighbors, n_bins_t, n_bins_p):
        self.rmax = rmax
        self.k = k
        self.num_neighbors = num_neighbors
        self.n_bins_t = n_bins_t
        self.n_bins_p = n_bins_p

    def bench_setup(self, N):
        # n = 4
        n = N
        (self.box, self.positions) = util.make_fcc(n, n, n)
        self.random_quats = np.random.rand(len(self.positions), 4)
        self.random_quats /= np.linalg.norm(self.random_quats,
                                            axis=1)[:, np.newaxis]

        self.bo = freud.environment.BondOrder(self.rmax, self.k,
                                              self.num_neighbors,
                                              self.n_bins_t, self.n_bins_p)

    def bench_run(self, N):
        self.bo.compute(self.box, self.positions, self.random_quats,
                        self.positions, self.random_quats)
