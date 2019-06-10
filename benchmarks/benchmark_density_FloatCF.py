import numpy as np
import freud
from benchmark import Benchmark
from benchmarker import run_benchmarks


class BenchmarkDensityFloatCF(Benchmark):
    def __init__(self, rmax, dr):
        self.rmax = rmax
        self.dr = dr

    def bench_setup(self, N):
        self.box_size = self.rmax*3.1
        np.random.seed(0)
        self.points = np.random.random_sample((N, 3)).astype(np.float32) \
            * self.box_size - self.box_size/2
        self.ang = np.random.random_sample((N)).astype(np.float64) - 0.5
        self.ocf = freud.density.FloatCF(self.rmax, self.dr)
        self.box = freud.box.Box.square(self.box_size)

    def bench_run(self, N):
        self.ocf.accumulate(self.box, self.points, self.ang)
        self.ocf.compute(self.box, self.points, self.ang)


def run():
    Ns = [1000, 10000]
    rmax = 10.0
    dr = 1.0
    name = 'freud.density.FloatCF'
    classobj = BenchmarkDensityFloatCF
    number = 100

    return run_benchmarks(name, Ns, number, classobj,
                          rmax=rmax, dr=dr)


if __name__ == '__main__':
    run()
