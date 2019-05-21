import numpy as np
import freud
from benchmark import Benchmark
from benchmarker import run_benchmarks


class BenchmarkMSDMSD(Benchmark):
    def __init__(self, L, mode):
        self.L = L
        self.mode = mode

    def bench_setup(self, N):
        box = freud.box.Box.cube(self.L)
        seed = 0
        np.random.seed(seed)
        N_frames = 10
        self.positions = np.asarray(np.random.uniform(-self.L/2, self.L/2,
                                                      (N_frames, N, 3)),
                                    dtype=np.float32)
        self.msd = freud.msd.MSD(box, self.mode)

    def bench_run(self, N):
        self.msd.compute(self.positions)


def run():
    Ns = [100, 500, 1000, 5000]
    print_stats = True
    number = 100
    name = 'freud.msd.MSD'

    kwargs = {"L": 10,
              "mode": "window"}

    return run_benchmarks(name, Ns, number, BenchmarkMSDMSD,
                          print_stats, **kwargs)


if __name__ == '__main__':
    run()
