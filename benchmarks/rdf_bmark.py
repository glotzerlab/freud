from freud import density
import numpy as np

import benchmark
import internal


class RDFBenchmark(benchmark.benchmark):

    def setup(self, N):
        nx = ny = int(np.round((N/4)**(1./3)))
        nz = N//4//nx//ny

        (self.box, self.points) = internal.make_fcc(nx, ny, nz, noise=1e-2)

    def run(self, N):
        rdf = density.RDF(5.0, 0.05)
        rdf.compute(self.box, self.points, self.points)


if __name__ == '__main__':
    times = RDFBenchmark().run_thread_scaling_benchmark(
        [4096, 16384, 65536], number=20)
    print(times)
