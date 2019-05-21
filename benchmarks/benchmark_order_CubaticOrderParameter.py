import numpy as np
import freud
from benchmark import Benchmark
from benchmarker import run_benchmarks
import rowan


class BenchmarkOrderCubaticOrderParameter(Benchmark):
    def __init__(self, t_initial, t_final, scale, n_replicates, seed):
        self.t_initial = t_initial
        self.t_final = t_final
        self.scale = scale
        self.n_replicates = n_replicates
        self.seed = seed

    def bench_setup(self, N):
        seed = 0
        np.random.seed(seed)
        self.orientations = rowan.random.random_sample((N, ))
        self.cop = freud.order.CubaticOrderParameter(self.t_initial,
                                                     self.t_final,
                                                     self.scale,
                                                     self.n_replicates,
                                                     self.seed)

    def bench_run(self, N):
        self.cop.compute(self.orientations)


def run():
    Ns = [1000, 10000, 100000, 500000]
    print_stats = True
    number = 100
    name = 'freud.order.CubaticOrderParameter'

    kwargs = {"t_initial": 5.0,
              "t_final": 0.001,
              "scale": 0.95,
              "n_replicates": 10,
              "seed": 0}

    return run_benchmarks(name, Ns, number,
                          BenchmarkOrderCubaticOrderParameter,
                          print_stats, **kwargs)


if __name__ == '__main__':
    run()
