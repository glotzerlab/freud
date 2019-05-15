import numpy as np
import git
import json
import os

from freud import locality, box
from benchmark_density_LocalDensity \
    import BenchmarkDensityLocalDensity
from benchmark_locality_NearestNeighbors \
    import BenchmarkLocalityNearestNeighbors
from benchmark_locality_LinkCell \
    import BenchmarkLocalityLinkCell
from benchmark_locality_AABBQuery \
    import BenchmarkLocalityAABBQuery


def benchmark_description(name, params):
    s = name + ": \n\t"
    s += ", ".join("{} = {}".format(str(k), str(v)) for k, v in params.items())
    return s


def do_some_benchmarks(name, Ns, number, classobj, print_stats, **kwargs):
    if print_stats:
        print(benchmark_description(name, kwargs))

    try:
        b = classobj(**kwargs)
    except TypeError:
        print("Wrong set of initizliation keyword \
            arguments for {}".format(str(classobj)))
        return {"name": name, "misc": "No result"}

    ssr = b.run_size_scaling_benchmark(Ns, number, print_stats)
    tsr = b.run_thread_scaling_benchmark(Ns, number, print_stats)

    if print_stats:
        print('\n ----------------')

    return {"name": name, "params": kwargs, "Ns": Ns,
            "size_scale": ssr, "thread_scale": tsr.tolist()}


def print_benchmark_results_in_human_readable_way(bresult):
    bdesc = benchmark_description(bresult["name"], bresult["params"])
    print(bdesc)
    for N, r in zip(bresult["Ns"], bresult["size_scale"]):
        print('{0:10d}'.format(N), end=': ')
        print('{0:8.3f} ms | {1:8.3f} ns per item'.format(
            r/1e-3, r/N/1e-9))


def save_benchmark_result(bresults):
    repo = git.Repo(search_parent_directories=True)
    this_script_path = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(this_script_path, "benchmark.txt")

    if os.path.exists(filename):
        with open(filename, 'r') as infile:
            data = json.load(infile)
            data[str(repo.head.commit)] = bresults
    else:
        data = {str(repo.head.commit): bresults}

    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=4)


def run_benchmarks():
    Ns = [1000, 10000, 100000]
    print_stats = False
    rcut = 0.5
    L = 10
    num_neighbors = 6
    number = 100

    name = 'freud.locality.NearestNeighbors'
    classobj = BenchmarkLocalityNearestNeighbors
    r1 = do_some_benchmarks(name, Ns, number, classobj, print_stats,
                            L=L, rcut=rcut, num_neighbors=num_neighbors)
    # print(r)
    # print_benchmark_results_in_human_readable_way(r)

    name = 'freud.locality.AABBQuery'
    classobj = BenchmarkLocalityNearestNeighbors
    r2 = do_some_benchmarks(name, Ns, 100, classobj, print_stats,
                            L=L, rcut=rcut)
    # print(r)
    save_benchmark_result([r1, r2])

    # name = 'freud.locality.LinkCell'
    # classobj = BenchmarkLocalityLinkCell
    # do_some_benchmarks(, Ns, 100, classobj, print_stats, L=L, rcut=rcut)

    # rcut = 1.0

    name = 'freud.locality.NearestNeighbors'
    classobj = BenchmarkLocalityNearestNeighbors
    do_some_benchmarks(name, Ns, number, classobj, print_stats,
                       L=L, rcut=rcut, num_neighbors=num_neighbors)
    # rcut = 10
    # nu = 1
    # name = 'freud.density.LocalDensity'
    # classobj = BenchmarkDensityLocalDensity
    # do_some_benchmarks(name, Ns, 100, classobj, print_stats,
    #                     nu=nu, rcut=rcut)


def compare_benchmarks(rev_this, rev_other):
    repo = git.Repo(search_parent_directories=True)
    rev_this = str(repo.commit(rev_this))
    rev_other = str(repo.commit(rev_other))
    # print(rev_this)
    # print(rev_other)

    this_script_path = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(this_script_path, "benchmark.txt")

    with open(filename, 'r') as infile:
        data = json.load(infile)

    rev_this_benchmark = data[rev_this]
    rev_other_benchmark = data[rev_other]

    print(rev_this_benchmark)
    print(rev_other_benchmark)


if __name__ == '__main__':
    # run_benchmarks()
    compare_benchmarks("HEAD", "HEAD")
