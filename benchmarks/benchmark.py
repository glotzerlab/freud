from __future__ import print_function
from __future__ import division

import cProfile
import os
import pstats
import sys
import multiprocessing
import numpy
import timeit

from freud import parallel


class Benchmark(object):
    """The freud Benchmark class for running benchmarks.

        Run a given method and print benchmark results

        The benchmark class sets up and runs benchmarks of given functions and/or
        methods and provides convenience routines for one time setup and multiple run
        invocations. Benchmarks may be run with simple timing,
        or with speedup numbers on multiple cores.

        It provides an N argument to set the benchmark size. Benchmarks that don't
        use an N can simply ignore it (and the user can set None). When N is present,
        timing results are normalized by N.

        Like the unit test module, it is designed to be an overridden class.
        Users must override :code:`bench_setup()` and :code:`bench_run()` to run their benchmark.
        :code:`bench_setup()` is called once at the start of every benchmark for one time setup.
    """  # noqa: E501
    def __init__(self):
        """Initialization

        The user should override this method with necessary parameters.
        """
        self.__N = None
        self.__t = 0

    def bench_setup(self, N):
        """Setup function for benchmark.

        The user must override this method to provide preliminary materials as member
        variables to be used in the :code:`bench_run()` method. The runtime of this method is
        not measured, and is run only once before :code:`bench_run()` is run.

        Args:
            N (int):
                Size of the input.
        """  # noqa: E501
        pass

    def bench_run(self, N):
        """Run function for benchmark.

        The user must override this method with lines that the user wants
        to measure the runtime.

        Args:
            N (int):
                Size of the input.
        """
        pass

    def bench_run_parallel(self, N, num_threads):
        """Run :code:`bench_run()` with :code:`num_threads` many threads.

        Args:
            N (int):
                Size of the input.
            num_threads (int):
                Number of threads.
        """
        parallel.setNumThreads(num_threads)
        self.bench_run(N)

    def setup_timer(self, N, num_threads):
        """Helper function to return a timeit.timer instance that uses
        :code:`bench_setup()` once to setup and measures the time of
        :code:`bench_run_parallel()` method.

        Args:
            N (int):
                Size of the input.
            num_threads (int):
                Number of threads.
        """
        setup = "self.bench_setup(N)"
        stmt = "self.bench_run_parallel(N, num_threads)"
        varmapping = {"self": self, "N": N, "num_threads": num_threads}
        return timeit.Timer(stmt=stmt, setup=setup, globals=varmapping)

    def run_benchmark(self, N=None, number=100, print_stats=False, repeat=1,
                      num_threads=0):
        """Perform the benchmark

        Args:
            N (int):
                Size of the input (Default value = None).
            number (int):
                Number of times to call bench_run() in case one run takes
                too little time to be significant (Default value = 100).
            print_stats (bool):
                Print stats to stdout (Default value = False).
            repeat (int):
                Number of times to repeat time measurement of
                bench_run() (Default value = 1).
            num_threads (int):
                Number of threads to use. If 0, use all avaliable threads
                (Default value = 0).

        Returns:
            int: The minimum time out of :code:`repeat` many
            calls to :code:`bench_run()`.
        """
        # initialize timer
        timer = self.setup_timer(N, num_threads)

        # run benchmark
        t = min(timer.repeat(repeat, number))

        # save results for later summarization
        self.__N = N
        self.__t = t / number
        if print_stats:
            self.print_stats()

        return self.__t

    def print_stats(self):
        """Print statistics on the last benchmark run

        Statistics are printed to stdout in a human readable form. Stats are
        printed for the results of the last call to run_benchmark.
        """
        if self.__N is not None:
            print('{0:8.3f} ms | {1:8.3f} ns per item'.format(
                self.__t/1e-3, self.__t/self.__N/1e-9))
        else:
            print('{0:8.3f} ms'.format(self.__t/1e-3))

    def run_profile(self, N=None, number=100):
        """Profile a benchmark run

        Args:
            N (int):
                Size of the input (Default value = None).
            number (int):
                Number of times to call bench_run() in case one run takes
                too little time to be significant (Default value = 100).

        Runs the benchmark and prints out a cProfile trace
        """
        # initilize timer
        timer = self.setup_timer(N)

        # run the profile
        cProfile.runctx("timer.timeit(number)",
                        globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()

    def run_size_scaling_benchmark(self, N_list, number=1000,
                                   print_stats=True, repeat=1):
        """Problem size scaling benchmark with all available threads.

        The size scaling benchmark autoscales number down linearly with
        problem size (down to a minimum of 1).

        Args:
            N_list (list of ints):
                N values to execute the problem at.
            number (int):
                Number of times to call bench_run() in case one run takes
                too little time to be significant (Default value = 100).
            print_stats (bool):
                Print stats to stdout (Default value = False).
            repeat (int):
                Number of times to repeat time measurement of
                bench_run() (Default value = 1).

        Returns:
            list of float: A list of average run() times
            following N (in seconds)
        """
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
            t = self.run_benchmark(N, current_number, print_stats, repeat)
            results.append(t)

        return results

    def run_thread_scaling_benchmark(self, N_list, number=1000,
                                     print_stats=True, repeat=1):
        """Thread scaling benchmark.

        The size scaling benchmark autoscales number down linearly with
        problem size (down to a minimum of 1).

        Args:
            N_list (list of ints):
                N values to execute the problem at.
            number (int):
                Number of times to call bench_run() in case one run takes
                too little time to be significant (Default value = 100).
            print_stats (bool):
                Print stats to stdout (Default value = False).
            repeat (int):
                Number of times to repeat time measurement of
                bench_run() (Default value = 1).

        Returns:
            :math:`\left(N_{cores}, len(N_{list})\right)` :class:`numpy.ndarray`:
                All the per iteration timings with respect to the number of
                cores used (in seconds).
        """  # noqa: E501
        if len(N_list) == 0:
            raise TypeError('N_list must be iterable')

        # compute benchmark size
        size = number*N_list[0]

        # print the header
        if print_stats:
            print('Threads ', end='')
            for N in N_list:
                print('{0:^22d}'.format(N), end=' | ')
            print()

        nproc_increment = int(os.environ.get('BENCHMARK_NPROC_INCREMENT', 1))
        nprocs = int(os.environ.get('BENCHMARK_NPROC',
                                    multiprocessing.cpu_count()))

        # loop over the cores
        times = numpy.zeros(shape=(nprocs+1, len(N_list)))

        for ncores in range(1, nprocs+1, nproc_increment):
            parallel.setNumThreads(ncores)

            if print_stats:
                print('{0:7d}'.format(ncores), end=' ')

            # loop over N and run the benchmarks
            for j, N in enumerate(N_list):
                current_number = max(int(size // N), 1)
                times[ncores, j] = self.run_benchmark(
                    N, number=current_number, print_stats=False, repeat=repeat,
                    num_threads=ncores)

                if print_stats:
                    speedup = times[1, j] / times[ncores, j]
                    print('{0:8.3f} ms {1:9.2f}x'.format(times[ncores, j]*1000,
                                                         speedup), end=' | ')
                    sys.stdout.flush()

            if print_stats:
                print()

        return times
