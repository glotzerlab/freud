import numpy as np
import freud
from benchmark import Benchmark
from benchmarker import run_benchmarks


class BenchmarkDensityRDF(Benchmark):
    def __init__(self, rmax, dr, rmin):
        self.rmax = rmax
        self.dr = dr
        self.rmin = rmin

    def bench_setup(self, N):
        self.box_size = self.rmax*3.1
        np.random.seed(0)
        self.points = np.random.random_sample((N, 3)).astype(np.float32) \
            * self.box_size - self.box_size/2
        self.rdf = freud.density.RDF(self.rmax, self.dr, r_min=self.rmin)
        self.box = freud.box.Box.cube(self.box_size)

    def bench_run(self, N):
        self.rdf.accumulate(self.box, self.points)
        self.rdf.compute(self.box, self.points)


def run():
    Ns = [1000, 10000]
    rmax = 10.0
    dr = 1.0
    rmin = 0
    number = 100
    name = 'freud.density.RDF'
    classobj = BenchmarkDensityRDF

    return run_benchmarks(name, Ns, number, classobj,
                          rmax=rmax, dr=dr, rmin=rmin)


if __name__ == '__main__':
    run()
