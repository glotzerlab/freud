from freud import box
from freud import locality
from freud import order
import random
import numpy
import time

import benchmark
import internal

class SteinhardtBenchmark(benchmark.benchmark):

    def setup(self, N):
        nx = ny = int(numpy.round((N/4)**(1./3)))
        nz = N//4//nx//ny

        (self.box, self.points) = internal.make_fcc(nx, ny, nz, noise=1e-2)

    def run(self, N):
        try:
            lc = locality.LinkCell(self.box, 2).computeCellList(self.box, self.points, exclude_ii=True)
            args = [lc.nlist, self.points]
        except:
            args = [self.points]
        stein = order.LocalQl(self.box, 2, 6)
        stein.compute(*args);

if __name__ == '__main__':
    times = SteinhardtBenchmark().run_thread_scaling_benchmark([4096, 16384, 65536], number=20)
    print(times)
