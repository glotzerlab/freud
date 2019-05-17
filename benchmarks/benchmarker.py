import numpy as np
import git
import json
import os
import argparse
import sys
import unittest
import importlib


# To add a new benchmark,
# 1) Make a file with name starting with benchmark_
# 2) Inherit Benchmark class
# 3) Implement run(on_circleci) method in the file

# filename with directory to save benchmark report
def get_report_filename(filename):
    this_script_path = os.path.dirname(os.path.abspath(__file__))
    report_filename = os.path.join(this_script_path, "reports", filename)
    return report_filename


# check if module exists and return the imported module if it exists
def try_importing(module):
    try:
        return importlib.import_module(module)
    except ImportError:
        print("{} does not exist and thus cannot"
              " be benchmarked".format(module))
        return None


# description string for benchmark run
def benchmark_desc(name, params):
    s = name + ": \n\t"
    s += ", ".join("{} = {}".format(str(k), str(v)) for k, v in params.items())
    return s


# param name: Name of benchmark run to print
# param Ns: List containing N values to run benchmark on
# param number: Number of times to run time
# param classobj: Benchmark class object to benchmark on
# param print_stats: Print stats if true
# param on_circleci: Limit thread number if ran on circle.ci
# param kwargs: Initializer variables for classobj
# return: Dictionary containing benchmark information
def do_some_benchmarks(name, Ns, number, classobj, print_stats,
                       on_circleci, **kwargs):
    if print_stats:
        print(benchmark_desc(name, kwargs))

    # initialize classobj instance
    try:
        b = classobj(**kwargs)
    except TypeError:
        print("Wrong set of initizliation keyword \
            arguments for {}".format(str(classobj)))
        return {"name": name, "misc": "No result"}

    # run benchmark with repeat
    repeat = 3
    ssr = b.run_size_scaling_benchmark(Ns, number, print_stats,
                                       repeat)
    tsr = b.run_thread_scaling_benchmark(Ns, number, print_stats,
                                         repeat, on_circleci)

    if print_stats:
        print('\n ----------------')

    return {"name": name, "params": kwargs, "Ns": Ns,
            "size_scale": {N: r for N, r in zip(Ns, ssr)},
            "thread_scale": tsr.tolist()}


# Print report
def main_report(args):
    filename = get_report_filename(args.filename)

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


# save benchmark result in current_dir/reports/filename
def save_benchmark_result(bresults, filename):
    repo = git.Repo(search_parent_directories=True)

    filename = get_report_filename(filename)
    this_script_path = os.path.dirname(os.path.abspath(__file__))

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


# save benchmark result in current_dir/reports/benchmark_comp.json
def save_comparison_result(rev_this, rev_other, slowers, fasters, sames):
    data = {"runtime": "{} / {}".format(rev_this, rev_other)}
    data["slowers"] = slowers
    data["fasters"] = fasters
    data["sames"] = sames
    filename = get_report_filename("benchmark_comp.json")
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=4)


# return list of all module names in the directory
# containing this script starting with name benchmark_
def list_benchmark_modules():
    import glob
    modules = glob.glob(os.path.join(os.path.dirname(__file__),
                                     "benchmark_*"))
    prefixdir = "benchmarks/"
    modules = [f[len(prefixdir):-3] for f in modules]
    return modules


# run benchmark on all modules in the directory
# containg this script starting with name benchmark_
def main_run(args):
    results = []
    modules = list_benchmark_modules()
    for m in modules:
        m = try_importing(m)
        if m:
            try:
                r = m.run(args.circleci)
                results.append(r)
            except AttributeError:
                print("Something is wrong with {}".format(m))

    save_benchmark_result(results, args.output)


# compare runtime of two commits and save the comparison
# result
# exit 1 if the runtime of rev_this is slower than
# the runtime of rev_other by more than the threshold ratio
# STRUCUTRE CAN BE IMPROVED the logic is simple
def main_compare(args):
    rt = args.rev_this
    ro = args.rev_other
    repo = git.Repo(search_parent_directories=True)
    rev_this = str(repo.commit(rt))
    rev_other = str(repo.commit(ro))

    filename = get_report_filename(args.filename)

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
                print("\nShowing runtime "
                      "{:6.6} ({:6.6}) / "
                      "{:6.6} ({:6.6})".format(ro, rev_other,
                                               rt, rev_this))
                print("")
                for N in this_res["Ns"]:
                    N = str(N)
                    this_t = this_res["size_scale"][N]
                    other_t = other_res["size_scale"][N]
                    ratio = other_t/this_t

                    print("N: {}, ratio: {:0.2f}".format(N, ratio))

                    info = {"name": this_res["name"],
                            "params": this_res["params"],
                            "N": N,
                            "ratio": ratio}

                    if ratio < 1:
                        print("\t{:6.6} is {:0.2f} times "
                              "slower than {:6.6}".format(rt, ratio, ro))
                        slowers.append(info)
                    if ratio > 1:
                        print("\t{:6.6} is {:0.2f} times "
                              "faster than {:6.6}".format(rt, ratio, ro))
                        fasters.append(info)
                    if ratio == 1:
                        print("\t{:6.6} and {:6.6} "
                              "have the same speed".format(rt, ro))
                        sames.append(info)

                num_threads = len(this_res["thread_scale"]) - 1
                for i in range(1, num_threads + 1):
                    for j, N in enumerate(this_res["Ns"]):
                        this_t = this_res["thread_scale"][i][j]
                        other_t = other_res["thread_scale"][i][j]
                        ratio = other_t/this_t

                        print("Threads: {}, N: {}, "
                              "ratio: {:0.2f}".format(str(i), N, ratio))

                        info = {"name": this_res["name"],
                                "params": this_res["params"],
                                "threads": i,
                                "N": N,
                                "ratio": ratio}

                        if ratio < 1:
                            print("\t{:6.6} is {:0.2f} times "
                                  "slower than {:6.6}".format(rt, ratio, ro))
                            slowers.append(info)
                        if ratio > 1:
                            print("\t{:6.6} is {:0.2f} times "
                                  "faster than {:6.6}".format(rt, ratio, ro))
                            fasters.append(info)
                        if ratio == 1:
                            print("\t{:6.6} and {:6.6} "
                                  "have the same speed".format(rt, ro))
                            sames.append(info)

                print('\n ----------------')

    save_comparison_result(rt, ro, slowers, fasters, sames)

    threshold = 0.70
    fail = False
    for info in slowers:
        if info["ratio"] < threshold:
            desc = benchmark_desc(info["name"], info["params"])
            print("TOO SLOW (beyond threshold of {})".format(threshold))
            print("\t" + desc)
            print("\t\tratio = {}".format(info["ratio"], threshold))
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
        '-c', '--circleci', action='store_true',
        help="Flag for running on circle.ci to fix thread number")
    parser_run.set_defaults(func=main_run)

    parser_report = subparsers.add_parser(
        name='report',
        description="Display results from previous runs.")
    parser_report.add_argument(
        'filename', default='benchmark.json', nargs='?',
        help="The collection that contains the benchmark data"
             "default='benchmark.json'.")

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
