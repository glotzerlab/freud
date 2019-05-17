import numpy as np
import numpy.testing as npt
import freud
import unittest
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
        self.points.flags['WRITEABLE'] = False
        self.rdf = freud.density.RDF(self.rmax, self.dr, rmin=self.rmin)

    def bench_run(self, N):
        self.rdf.accumulate(freud.box.Box.cube(self.box_size), self.points)
        self.rdf.compute(freud.box.Box.cube(self.box_size), self.points)


def run():
    Ns = [1000, 10000]
    rmax = 10.0
    dr = 1.0
    rmin = 0
    number = 100
    name = 'freud.density.RDF'
    classobj = BenchmarkDensityRDF
    print_stats = True
    return run_benchmarks(name, Ns, number, classobj, print_stats,
                          rmax=rmax, dr=dr, rmin=rmin)
