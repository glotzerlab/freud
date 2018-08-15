from __future__ import print_function
from __future__ import division

import time
import cProfile
import os
import pstats
import sys
import multiprocessing
import numpy

from freud import parallel

# Run a given method and print benchmark results
#
# The benchmark class sets up and runs benchmarks of given functions and/or
# methods and provides convenience routines for one time setup and multiple run
# invocations. Benchmarks may be run with profiling enabled, with simple
# timing, or with speedup numbers on multiple cores.
#
# It provides an N argument to set the benchmark size. Benchmarks that don't
# use an N can simply ignore it (and the user can set None). When N is present,
# timing results are normalized by N.
#
# Like the unit test module, it is designed to be an overridden class.
# Users must override run() to run their benchmark.
# setup() is called once at the start of every benchmark for one time setup.


class benchmark(object):

    def __init__(self):
        self.__N = None
        self.__t = 0

    # Override this method to provide one-time setup for a benchmark
    def setup(self, N):
        pass

    # Override this method to run the benchmark
    def run(self, N):
        pass

    # Perform the benchmark
    #
    # \param N Problem passed to setup() and run.
    # \param number Number of times to call run()
    # \param print_stats Print stats to stdout
    # \returns The average time for each call to run()
    #
    def run_benchmark(self, N=None, number=100, print_stats=False):
        # setup
        self.setup(N)
        self.run(N)

        # run the benchmark
        start = time.time()
        for i in range(0, number):
            self.run(N)
        end = time.time()

        # save results for later summarization
        self.__N = N
        self.__t = (end - start) / number
        if print_stats:
            self.print_stats()

        return self.__t

    # Print statistics on the last benchmark run
    #
    # Statistics are printed to stdout in a human readable form. Stats are
    # printed for the results of the last call to run_benchmark.
    def print_stats(self):
        if self.__N is not None:
            print('{0:8.3f} ms | {1:8.3f} ns per item'.format(
                self.__t/1e-3, self.__t/self.__N/1e-9))
        else:
            print('{0:8.3f} ms'.format(self.__t/1e-3))

    # Profile a benchmark run
    #
    # \param N Problem passed to setup() and run.
    # \param number Number of times to call run()
    #
    # Runs the benchmark and prints out a cProfile trace
    def run_profile(self, N=None, number=100):
        # setup
        self.setup(N)
        self.run(N)

        # run the profile
        cProfile.runctx("for i in range(0,number): self.run(N);",
                        globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()

    # System size scaling benchmark
    #
    # \param N_list list of the N values to execute the problem at
    # \param number Number of times to call run() at each problem size
    # \param print_stats Print the stats to stdout
    # \returns A list of average run() times following N (in seconds)
    #
    # \note The size scaling benchmark autoscales number down linearly with
    #       problem size (down to a minimum of 1).
    #
    def run_size_scaling_benchmark(self, N_list, number=1000,
                                   print_stats=True):
        if len(N_list) == 0:
            raise TypeError('N_list must be iterable')

        # compute benchmark size
        size = number*N_list[0]

        # loop over N and run the benchmarks
        results = []
        for N in N_list:
            if print_stats:
                print('{0:10d}'.format(N), end=': ')
                sys.stdout.flush()

            current_number = max(int(size // N), 1)
            t = self.run_benchmark(N, current_number, print_stats)
            results.append(t)

        return results

    # Thread scaling benchmark
    #
    # \param N_list list of the N values to execute the problem at.
    # \param number Number of times to call run() at each problem size.
    # \param print_stats Print the stats to stdout.
    # \returns A numpy array Ncores x len(N_list) with all the per iteration
    #          timings (in seconds).
    #
    # \note The size scaling benchmark autoscales number down linearly with
    #       problem size (down to a minimum of 1).
    #
    def run_thread_scaling_benchmark(self, N_list, number=1000,
                                     print_stats=True):
        if len(N_list) == 0:
            raise TypeError('N_list must be iterable')

        # compute benchmark size
        size = number*N_list[0]

        # print the header
        if print_stats:
            print('Threads ', end='')
            for N in N_list:
                print('{0:10d}'.format(N), end=' | ')
            print()

        nproc_increment = int(os.environ.get('BENCHMARK_NPROC_INCREMENT', 1))
        # loop over the cores
        times = numpy.zeros(shape=(multiprocessing.cpu_count()+1, len(N_list)))
        for ncores in range(1, multiprocessing.cpu_count()+1, nproc_increment):
            parallel.setNumThreads(ncores)

            if print_stats:
                print('{0:7d}'.format(ncores), end=' ')

            # loop over N and run the benchmarks
            for j, N in enumerate(N_list):
                current_number = max(int(size // N), 1)
                times[ncores, j] = self.run_benchmark(
                    N, number=current_number, print_stats=False)

                if print_stats:
                    speedup = times[1, j] / times[ncores, j]
                    print('{0:9.2f}x'.format(speedup), end=' | ')
                    # print('{0:10.2f}'.format(t), end=' | ')
                    sys.stdout.flush()

            print()

        return times
