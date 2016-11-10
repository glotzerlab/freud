from __future__ import print_function
from __future__ import division

from freud import density, box
from benchmark import benchmark
import numpy
import math

class benchmark_local_density(benchmark):
    def __init__(self, nu, r_cut):
        self.nu = nu;
        self.r_cut = r_cut;
    
    def setup(self, N):
        # setup a 2D system of random points
        box_size = math.sqrt(N*self.nu);
        self.pos = numpy.random.random_sample((N,3)).astype(numpy.float32)*box_size - box_size/2
        self.pos[:,2] = 0;
        self.ld = density.LocalDensity(self.r_cut, 1, 1);
    
    def run(self, N):
        box_size = math.sqrt(N*self.nu);
        self.ld.compute(box.Box.square(box_size), self.pos);


if __name__ == '__main__':
    print('local_density');
    b = benchmark_local_density(nu=1, r_cut=10);
    b.run_size_scaling_benchmark([1000, 10000, 100000, 1000000], number=100);
    b.run_thread_scaling_benchmark([1000, 10000, 100000, 1000000], number=100);
    
    #print('\n profiling');
    #b.run_profile(100000);

    print('\n ----------------');

    # print('grayscale');
    # b = benchmark_grayscale();
    # b.run_size_scaling_benchmark([100, 1000, 10000, 100000, 1000000], number=10000);
    
    # #print('\n profiling');
    # #b.run_profile(100000);

    # print('\n ----------------');
