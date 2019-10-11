import numpy as np
import freud
from benchmark import Benchmark
from benchmarker import run_benchmarks


class BenchmarkClusterCluster(Benchmark):
    def __init__(self, L, rcut):
        self.L = L
        self.rcut = rcut

    def bench_setup(self, N):
        self.box = freud.box.Box.cube(self.L)
        seed = 0
        np.random.seed(seed)
        self.positions = np.random.uniform(-self.L/2, self.L/2, (N, 3))

    def bench_run(self, N):
        clust = freud.cluster.Cluster()
        clust.compute((self.box, self.positions), keys=np.arange(N),
                      neighbors={'r_max': self.rcut})


def run():
    Ns = [1000, 5000, 10000]
    rcut = 1.0
    L = 10
    name = 'freud.cluster.Cluster'
    classobj = BenchmarkClusterCluster
    number = 100

    return run_benchmarks(name, Ns, number, classobj,
                          L=L, rcut=rcut)


if __name__ == '__main__':
    run()
