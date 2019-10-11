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
        seed = 0
        np.random.seed(seed)
        self.positions = np.random.uniform(-self.L/2, self.L/2, (N, 3))
        self.pbuff = freud.box.PeriodicBuffer()

    def bench_run(self, N):
        box = freud.box.Box.cube(self.L)
        self.pbuff.compute((box, self.positions), buffer=self.buffer,
                           images=self.images)


def run():
    Ns = [1000, 5000, 10000]
    number = 100
    name = 'freud.box.ParticleBuffer'

    L = 10
    buf = 2
    images = True
    return run_benchmarks(name, Ns, number, BenchmarkParticleBuffer,
                          L=L, buf=buf, images=images)


if __name__ == '__main__':
    run()
