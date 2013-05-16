from __future__ import print_function
from __future__ import division

from freud import viz
from benchmark import benchmark
import numpy

class benchmark_hsv(benchmark):
    def setup(self, N):
        self.input = numpy.random.random((N,)).astype(numpy.float32);
    
    def run(self, N):
        viz.colormap.hsv(self.input);

class benchmark_grayscale(benchmark):
    def setup(self, N):
        self.input = numpy.random.random((N,)).astype(numpy.float32);
    
    def run(self, N):
        viz.colormap.grayscale(self.input);

if __name__ == '__main__':
    print('hsv');
    b = benchmark_hsv();
    b.run_size_scaling_benchmark([100, 1000, 10000, 100000, 1000000], number=10000);
    
    #print('\n profiling');
    #b.run_profile(100000);

    print('\n ----------------');

    # print('grayscale');
    # b = benchmark_grayscale();
    # b.run_size_scaling_benchmark([100, 1000, 10000, 100000, 1000000], number=10000);
    
    # #print('\n profiling');
    # #b.run_profile(100000);

    # print('\n ----------------');
