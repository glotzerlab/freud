import numpy as np
import freud
from benchmark import Benchmark
from benchmarker import run_benchmarks
import rowan


class BenchmarkOrderRotationalAutocorrelation(Benchmark):
    def __init__(self, _I):
        self._I = _I

    def bench_setup(self, N):
        seed = 0
        np.random.seed(seed)
        self.orientations = rowan.random.random_sample((N, ))
        self.ra = freud.order.RotationalAutocorrelation(self._I)

    def bench_run(self, N):
        self.ra.compute(self.orientations, self.orientations)


def run():
    Ns = [1000, 5000, 10000]
    number = 100
    name = 'freud.order.RotationalAutocorrelation'

    kwargs = {"_I": 2}

    return run_benchmarks(name, Ns, number,
                          BenchmarkOrderRotationalAutocorrelation,
                          **kwargs)


if __name__ == '__main__':
    run()
