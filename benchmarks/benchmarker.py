import numpy as np
import git
import json
import os
import argparse
import sys
import unittest
import importlib


def try_importing(module):
    try:
        return importlib.import_module(module)
    except ImportError:
        print("{} does not exist and thus cannot"
              " be benchmarked".format(module))
        return None


def benchmark_desc(name, params):
    s = name + ": \n\t"
    s += ", ".join("{} = {}".format(str(k), str(v)) for k, v in params.items())
    return s


def do_some_benchmarks(name, Ns, number, classobj, print_stats, **kwargs):
    if print_stats:
        print(benchmark_desc(name, kwargs))

    try:
        b = classobj(**kwargs)
    except TypeError:
        print("Wrong set of initizliation keyword \
            arguments for {}".format(str(classobj)))
        return {"name": name, "misc": "No result"}

    repeat = 1
    ssr = b.run_size_scaling_benchmark(Ns, number, print_stats, repeat)
    tsr = b.run_thread_scaling_benchmark(Ns, number, print_stats, repeat)

    if print_stats:
        print('\n ----------------')

    return {"name": name, "params": kwargs, "Ns": Ns,
            "size_scale": {N: r for N, r in zip(Ns, ssr)},
            "thread_scale": tsr.tolist()}


def main_report(args):
    this_script_path = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(this_script_path, "reports", args.filename)

    with open(filename, 'r') as infile:
        data = json.load(infile)
    for commit in data:
        print("Commit {}:".format(commit))
        print_benchmark_results_in_human_readable_way(data[commit])


def print_benchmark_results_in_human_readable_way(data):
    for bresult in data:
        bdesc = benchmark_desc(bresult["name"], bresult["params"])
        print(bdesc)
        for N, r in bresult["size_scale"].items():
            N = int(N)
            r = float(r)
            print('{0:10d}'.format(N), end=': ')
            print('{0:8.3f} ms | {1:8.3f} ns per item'.format(
                float(r)/1e-3, float(r)/int(N)/1e-9))


def save_benchmark_result(bresults, filename):
    repo = git.Repo(search_parent_directories=True)
    this_script_path = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(this_script_path, "reports", filename)

    if not os.path.exists(os.path.join(this_script_path, "reports")):
        os.mkdir(os.path.join(this_script_path, "reports"))

    if os.path.exists(filename):
        with open(filename, 'r') as infile:
            data = json.load(infile)
            data[str(repo.head.commit)] = bresults
    else:
        data = {str(repo.head.commit): bresults}

    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=4)


def list_benchmark_modules():
    import glob
    modules = glob.glob(os.path.join(os.path.dirname(__file__),
                                     "benchmark_*"))
    modules = [f[:-3] for f in modules]
    return modules


def main_run(args):
    results = []
    modules = list_benchmark_modules()

    for m in modules:
        m = try_importing(m)
        results.append(m.run())

    save_benchmark_result(results, args.output)


def main_compare(args):
    rt = args.rev_this
    ro = args.rev_other
    repo = git.Repo(search_parent_directories=True)
    rev_this = str(repo.commit(rt))
    rev_other = str(repo.commit(ro))

    this_script_path = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(this_script_path, "reports", args.filename)

    with open(filename, 'r') as infile:
        data = json.load(infile)

    rev_this_benchmark = data[rev_this]
    rev_other_benchmark = data[rev_other]

    slowers = []
    fasters = []
    sames = []

    for this_res in rev_this_benchmark:
        for other_res in rev_other_benchmark:
            if this_res["name"] == other_res["name"] \
                    and this_res["params"] == other_res["params"]:
                print(benchmark_desc(this_res["name"],
                                     this_res["params"]))
                print("Showing runtime "
                      "{:6.6} ({:6.6}) / "
                      "{:6.6} ({:6.6})".format(rt, rev_this, ro, rev_other))
                for N in this_res["Ns"]:
                    N = str(N)
                    this_t = this_res["size_scale"][N]
                    other_t = other_res["size_scale"][N]
                    ratio = this_t/other_t
                    print("N: {}, ratio: {:0.2f}".format(N, ratio))
                    info = {"name": this_res["name"],
                            "params": this_res["params"],
                            "N": N,
                            "ratio": ratio}
                    if ratio > 1:
                        print("{:6.6} is slower than {:6.6}".format(rt, ro))
                        slowers.append(info)
                    if ratio < 1:
                        print("{:6.6} is faster than {:6.6}".format(rt, ro))
                        fasters.append(info)
                    if ratio == 1:
                        print("{:6.6} and {:6.6} "
                              "have the same speed".format(rt, ro))
                        sames.append(info)
                print('\n ----------------')

    threshold = 1.2
    fail = False
    for info in slowers:
        if info["ratio"] > threshold:
            desc = benchmark_desc(info["name"], info["params"])
            print("{} too slow".format(desc))
            print("ratio = {} > threshold = {}".format(info["ratio"],
                                                       threshold))
            fail = True
    if fail:
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Test the runtime performance of freud")
    subparsers = parser.add_subparsers()

    parser_run = subparsers.add_parser(
        name='run',
        description="Execute performance tests in various categories for "
                    "specific data space sizes (N).")
    parser_run.add_argument(
        '-o', '--output', nargs='?', default='benchmark.json',
        help="Specify which collection file to store results "
             "to or '-' for None, "
             "default='benchmark.json'.")
    parser_run.add_argument(
        '-N', type=int, default=[1000, 10000, 100000], nargs='+',
        help="The number of data/ state points within the "
             "benchmarked project. "
             "The default size is 100. Specify more than "
             "one value to test multiple "
             "different size sequentally.")
    parser_run.add_argument(
        '-p', '--profile', action='store_true',
        help="Activate profiling (Results should not be used for reporting.")
    parser_run.set_defaults(func=main_run)

    parser_report = subparsers.add_parser(
        name='report',
        description="Display results from previous runs.")
    parser_report.add_argument(
        'filename', default='benchmark.json', nargs='?',
        help="The collection that contains the benchmark data"
             "default='benchmark.json'.")
    parser_report.add_argument(
        '-f', '--filter', type=str,
        help="Select a subset of the data.")
    parser_report.set_defaults(func=main_report)

    parser_compare = subparsers.add_parser(
        name='compare',
        description="Compare performance between two "
                    "git-revisions of this repository. "
                    "For example, to compare the current revision "
                    "(HEAD) with the "
                    "'master' branch revision, execute `{} compare "
                    "master HEAD`. In this specific "
                    "case one could omit both arguments, since 'master'"
                    " and 'HEAD' are the two "
                    "default arguments.".format(sys.argv[0]))
    parser_compare.add_argument(
        'rev_other', default='master', nargs='?',
        help="The git revision to compare against. "
             "Valid arguments are  for example "
             "a branch name, a tag, a specific commit id, "
             "or 'HEAD', defaults to 'master'.")
    parser_compare.add_argument(
        'rev_this', default='HEAD', nargs='?',
        help="The git revision that is benchmarked. "
             "Valid arguments are  for example "
             "a branch name, a tag, a specific commit id, "
             "or 'HEAD', defaults to 'HEAD'.")
    parser_compare.add_argument(
        '--filename', default='benchmark.json', nargs='?',
        help="The collection that contains the benchmark data"
             "default='benchmark.json'.")
    parser_compare.add_argument(
        '-f', '--fail-above',
        type=float,
        help="Exit with error code in case that the runtime ratio of "
             "the worst tested category between this and the other revision "
             "is above this value.")
    parser_compare.set_defaults(func=main_compare)

    args = parser.parse_args()
    args.func(args)
