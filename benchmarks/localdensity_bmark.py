from freud import box
from freud import density
from freud import locality
import random
import numpy
import time

import benchmark
import internal

class LocalDensityBenchmark(benchmark.benchmark):

    def setup(self, N):
        nx = ny = int(numpy.round((N/4)**(1./3)))
        nz = N//4//nx//ny

        (self.box, self.points) = internal.make_fcc(nx, ny, nz, noise=1e-2)

    def run(self, N):
        try:
            lc = locality.LinkCell(self.box, 5).computeCellList(self.box, self.points, exclude_ii=True)
            args = [self.box, lc.nlist, self.points]
        except:
            args = [self.box, self.points]
        dens = density.LocalDensity(5, 1, 1)
        dens.compute(*args);

if __name__ == '__main__':
    times = LocalDensityBenchmark().run_thread_scaling_benchmark([4096, 16384, 65536], number=20)
    print(times)
