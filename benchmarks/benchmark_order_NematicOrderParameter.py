import numpy as np
import freud
from benchmark import Benchmark
from benchmarker import run_benchmarks
import rowan


class BenchmarkOrderNematicOrderParameter(Benchmark):
    def __init__(self, u):
        self.u = u

    def bench_setup(self, N):
        seed = 0
        np.random.seed(seed)
        self.orientations = rowan.random.random_sample((N, ))
        self.nop = freud.order.NematicOrderParameter(np.array(self.u))

    def bench_run(self, N):
        self.nop.compute(self.orientations)


def run():
    Ns = [1000, 5000, 10000]
    print_stats = True
    number = 100
    name = 'freud.order.NematicOrderParameter'

    kwargs = {"u": [1, 0, 0]}

    return run_benchmarks(name, Ns, number,
                          BenchmarkOrderNematicOrderParameter,
                          print_stats, **kwargs)


if __name__ == '__main__':
    run()
