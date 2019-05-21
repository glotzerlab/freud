import numpy as np
import freud
from benchmark import Benchmark
from benchmarker import run_benchmarks


class BenchmarkParticleBuffer(Benchmark):
    def __init__(self, L, buf, images):
        self.L = L
        self.buffer = buf
        self.images = images

    def bench_setup(self, N):
        fbox = freud.box.Box.cube(self.L)
        seed = 0
        np.random.seed(seed)
        self.positions = np.random.uniform(-self.L/2, self.L/2, (N, 3))
        self.pbuff = freud.box.ParticleBuffer(fbox)

    def bench_run(self, N):
        self.pbuff.compute(self.positions, buffer=self.buffer,
                           images=self.images)


def run():
    Ns = [1000, 10000, 100000, 1000000]
    print_stats = True
    number = 100
    name = 'freud.box.ParticleBuffer'

    # L = 10
    # buf = L*0.5
    # images = False
    L = 10
    buf = 2
    images = True
    return run_benchmarks(name, Ns, number, BenchmarkParticleBuffer,
                          print_stats, L=L, buf=buf, images=images)


if __name__ == '__main__':
    run()
