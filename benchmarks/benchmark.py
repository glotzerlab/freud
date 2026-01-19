# Copyright (c) 2010-2026 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import cProfile
import multiprocessing
import os
import pstats
import sys
import timeit

import numpy

import freud


class Benchmark:
    """The freud Benchmark class for running benchmarks and showing results.

    The benchmark class sets up and runs benchmarks of given functions and/or
    methods and provides convenience routines for one time setup and multiple
    run invocations. Benchmarks may be run with simple timing, or with speedup
    numbers on multiple cores.

    It provides an N argument to set the benchmark size. Benchmarks that don't
    use an N can simply ignore it (and the user can set :code:`None`). When N
    is present, timing results are normalized by N.

    Like the :code:`unittest` module, it is designed to be an overridden class.
    Users must override :py:meth:`~.bench_setup` and :py:meth:`~.bench_run` to
    run their benchmark. :py:meth:`~.bench_setup` is called once at the start
    of every benchmark for one-time setup.
    """

    def __init__(self):
        """Initialize benchmark.

        The user should override this method with necessary parameters.
        """
        self._N = None
        self._t = 0

    def bench_setup(self, N):
        """Setup function for benchmark.

        The user must override this method to provide preliminary materials as
        member variables to be used in the :py:meth:`~.bench_run` method. The
        runtime of this method is not measured, and is run only once before
        :py:meth:`~.bench_run` is run.

        Args:
            N (int):
                Size of the input.
        """
        pass

    def bench_run(self, N):
        """Run function for benchmark.

        This method is overridden with the code to benchmark.

        Args:
            N (int):
                Size of the input.
        """
        pass

    def bench_run_parallel(self, N, num_threads):
        """Run :py:meth:`~.bench_run` with :code:`num_threads` threads.

        Args:
            N (int):
                Size of the input.
            num_threads (int):
                Number of threads.
        """
        with freud.parallel.NumThreads(num_threads):
            self.bench_run(N)

    def setup_timer(self, N, num_threads):
        """Helper function to return a timeit.timer instance that uses
        :py:meth:`~.bench_setup` once to setup and measures the time of
        :py:meth:`~.bench_run_parallel` method.

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

    def run_benchmark(
        self, N=None, number=100, print_stats=False, repeat=1, num_threads=0
    ):
        """Perform the benchmark.

        Args:
            N (int):
                Size of the input (Default value = None).
            number (int):
                Number of times to call bench_run() in case one run takes
                too little time to be significant (Default value = 100).
            print_stats (bool):
                Print stats to stdout (Default value = :code:`False`).
            repeat (int):
                Number of times to repeat the measurement of
                :py:meth:`~.bench_run` (Default value = 1).
            num_threads (int):
                Number of threads to use. If 0, use all avaliable threads
                (Default value = 0).

        Returns:
            float: The minimum time out of :code:`repeat` many calls to
                :py:meth:`~.bench_run`.
        """
        # Initialize timer
        timer = self.setup_timer(N, num_threads)

        # Run benchmark
        t = numpy.median(timer.repeat(repeat, number))

        # Save results for later summarization
        self._N = N
        self._t = t / number
        if print_stats:
            self.print_stats()

        return self._t

    def print_stats(self):
        """Print statistics from the last benchmark run.

        Statistics are printed to stdout in a human readable form. Stats are
        printed for the results of the last call to :py:meth:`~.run_benchmark`.
        """
        if self._N is not None:
            print(
                f"{self._t / 1e-3:8.3f} ms | {self._t / self._N / 1e-9:8.3f}"
                " ns per item"
            )
        else:
            print(f"{self._t / 1e-3:8.3f} ms")

    def run_profile(self, N=None, number=100):
        """Profile a benchmark run.

        Args:
            N (int):
                Size of the input (Default value = :code:`None`).
            number (int):
                Number of times to call :py:meth:`~.bench_run` in case one run
                takes too little time to be significant (Default value = 100).

        Runs the benchmark and prints out a cProfile trace.
        """
        # Initialize timer
        timer = self.setup_timer(N)

        # Run the profile
        cProfile.runctx("timer.timeit(number)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()

    def run_size_scaling_benchmark(
        self, N_list, number=1000, print_stats=True, repeat=1
    ):
        """Problem size scaling benchmark with all available threads.

        The size scaling benchmark autoscales number down linearly with problem
        size (down to a minimum of 1).

        Args:
            N_list (list of ints):
                List of problem sizes :math:`N` to run.
            number (int):
                Number of times to call :py:meth:`~.bench_run` in case one run
                takes too little time to be significant (Default value = 1000).
            print_stats (bool):
                Print stats to stdout (Default value = :code:`False`).
            repeat (int):
                Number of times to repeat time measurement of
                :py:meth:`~.bench_run` (Default value = 1).

        Returns:
            list of float: A list of average runtimes following N (in seconds)
        """
        if len(N_list) == 0:
            msg = "N_list must be iterable"
            raise TypeError(msg)

        # Compute benchmark size
        size = number * N_list[0]

        # Loop over N and run the benchmarks
        results = []
        for N in N_list:
            if print_stats:
                print(f"{N:10d}", end=": ")
                sys.stdout.flush()

            current_number = max(int(size // N), 1)
            t = self.run_benchmark(N, current_number, print_stats, repeat)
            results.append(t)

        return results

    def run_thread_scaling_benchmark(
        self, N_list, number=1000, print_stats=True, repeat=1
    ):
        """Thread scaling benchmark.

        The size scaling benchmark autoscales number down linearly with problem
        size (down to a minimum of 1).

        Args:
            N_list (list of ints):
                List of problem sizes :math:`N` to run.
            number (int):
                Number of times to call :py:meth:`~.bench_run` in case one run
                takes too little time to be significant (Default value = 1000).
            print_stats (bool):
                Print stats to stdout (Default value = :code:`False`).
            repeat (int):
                Number of times to repeat time measurement of
                :py:meth:`~.bench_run` (Default value = 1).

        Returns:
            :math:`(N_{cores}, len(N_{list}))` :class:`numpy.ndarray`:
                All the per iteration timings with respect to the number of
                cores used (in seconds).
        """
        if len(N_list) == 0:
            msg = "N_list must be iterable"
            raise TypeError(msg)

        # compute benchmark size
        size = number * N_list[0]

        # print the header
        if print_stats:
            print("Threads ", end="")
            for N in N_list:
                print(f"{N:^22d}", end=" | ")
            print()

        nproc_increment = int(os.environ.get("BENCHMARK_NPROC_INCREMENT", "1"))
        nprocs = int(os.environ.get("BENCHMARK_NPROC", multiprocessing.cpu_count()))

        # loop over the cores
        times = numpy.zeros(shape=(nprocs + 1, len(N_list)))

        for ncores in range(1, nprocs + 1, nproc_increment):
            if print_stats:
                print(f"{ncores:7d}", end=" ")

            # Loop over N and run the benchmarks
            for j, N in enumerate(N_list):
                current_number = max(int(size // N), 1)

                with freud.parallel.NumThreads(ncores):
                    times[ncores, j] = self.run_benchmark(
                        N,
                        number=current_number,
                        print_stats=False,
                        repeat=repeat,
                        num_threads=ncores,
                    )

                if print_stats:
                    speedup = times[1, j] / times[ncores, j]
                    print(
                        f"{times[ncores, j] * 1000:8.3f} ms {speedup:9.2f}x",
                        end=" | ",
                    )
                    sys.stdout.flush()

            if print_stats:
                print()

        return times
