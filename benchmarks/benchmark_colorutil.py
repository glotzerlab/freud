from __future__ import print_function
from __future__ import division

from freud import viz, parallel
from benchmark import benchmark
import numpy

class benchmark_linearToSRGBA(benchmark):
    def setup(self, N):
        self.input = numpy.random.random((N,4)).astype(numpy.float32);
    
    def run(self, N):
        viz.colorutil.linearToSRGBA(self.input);

class benchmark_sRGBAtoLinear(benchmark):
    def setup(self, N):
        self.input = numpy.random.random((N,4)).astype(numpy.float32);
    
    def run(self, N):
        viz.colorutil.sRGBAtoLinear(self.input);

if __name__ == '__main__':
    print('linearToSRGBA');
    b = benchmark_linearToSRGBA();
    b.run_size_scaling_benchmark([100, 1000, 10000, 100000, 1000000], number=10000);
    print();
    b.run_thread_scaling_benchmark([100, 1000, 10000, 100000, 1000000], number=10000);
    
    #print('\n profiling');
    #b.run_profile(100000);

    print('\n ----------------');

    print('sRGBAtoLinear');
    b = benchmark_sRGBAtoLinear();
    b.run_size_scaling_benchmark([100, 1000, 10000, 100000, 1000000], number=10000);
    print()
    b.run_thread_scaling_benchmark([100, 1000, 10000, 100000, 1000000], number=10000);
    
    #print('\n profiling');
    #b.run_profile(100000);

    print('\n ----------------');
