# Copyright (c) 2010-2024 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import argparse
import importlib
import json
import os
import sys

import git


def get_report_filename(filename):
    """Function to get the directory to save benchmark report.

    Args:
        filename (str): Name of the file to store the benchmark report.

    Returns:
        str: Directory to save benchmark report.

    """
    this_script_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(this_script_path, "reports", filename)


def try_importing(module):
    """Function to check if a file exists and import if it does.

    Args:
        module (str): Name of the file in the current directory.

    Returns:
        Module if import successful, None otherwise.

    """
    try:
        return importlib.import_module(module)
    except ImportError:
        print(f"{module} does not exist and thus cannot be benchmarked")
        return None


def benchmark_desc(name, params):
    """Function to generate description of benchmark run.

    Args:
        name (str): Name of the benchmark.
        params (dict): Dictionary containing the parameters
            of the benchmark run.

    Returns:
        str: String with name and parameters.

    """
    s = name + ": \n\t"
    s += ", ".join(f"{k!s} = {v!s}" for k, v in params.items())
    return s


def run_benchmarks(name, Ns, number, classobj, print_stats=True, **kwargs):
    """Function to run benchmark.

    Args:
        name (str): Name of the benchmark.
        Ns (list of int): List of N values to run benchmark.
        number (int): Number of times to run to measure the time.
        classobj (Benchmark): Benchmark class to run benchmark.
        print_stats (bool): Print stats if true.
        **kwargs: Initializer variables for classobj.

    Returns:
        dict: Dictionary object containing benchmark results.

    """
    if print_stats:
        print(benchmark_desc(name, kwargs))

    # initialize classobj instance
    try:
        b = classobj(**kwargs)
    except TypeError:
        print(
            f"Wrong set of initialization keyword \
            arguments for {classobj!s}"
        )
        return {"name": name, "misc": "No result"}

    # run benchmark with repeat
    repeat = 5
    ssr = b.run_size_scaling_benchmark(Ns, number, print_stats, repeat)
    tsr = b.run_thread_scaling_benchmark(Ns, number, print_stats, repeat)

    if print_stats:
        print("\n ----------------")

    return {
        "name": name,
        "params": kwargs,
        "Ns": Ns,
        "size_scale": {N: r for N, r in zip(Ns, ssr)},
        "thread_scale": tsr.tolist(),
    }


def main_report(args):
    """Function to print report.

    Args:
        args (argparse.ArgumentParser): Argument parser.

    Returns:
        None.

    """
    filename = get_report_filename(args.filename)

    with open(filename) as infile:
        data = json.load(infile)
    for commit in data:
        print(f"Commit {commit}:")
        print_benchmark_results_in_human_readable_way(data[commit])
        print("\n ----------------")


def print_benchmark_results_in_human_readable_way(data):
    """Helper function to print nicely.

    Args:
        data (dict): Dictionary containing benchmark results.

    Returns:
        None.

    """
    for bresult in data:
        bdesc = benchmark_desc(bresult["name"], bresult["params"])
        print(bdesc)

        # print size scaling benchmark
        for N_loop, r_loop in bresult["size_scale"].items():
            N = int(N_loop)
            r = float(r_loop)
            print(f"{N:10d}", end=": ")
            print(
                f"{float(r) / 1e-3:8.3f} ms | {float(r) / int(N) / 1e-9:8.3f}"
                " ns per item"
            )

        # print thread scaling benchmark
        print("Threads ", end="")
        for N in bresult["Ns"]:
            print(f"{N:10d}", end=" | ")
        print()
        times = bresult["thread_scale"]
        num_threads = len(times) - 1
        for i in range(1, num_threads + 1):
            print(f"{i:7d}", end=" ")
            for j, _N in enumerate(bresult["Ns"]):
                speedup = times[1][j] / times[i][j]
                print(f"{speedup:9.2f}x", end=" | ")
            print()


def save_benchmark_result(bresults, filename):
    """Function to save benchmark result.

    Result saved in directory_of_this_script/reports/filename.

       If file already exists, append the result.

    Args:
        bresults (list of dict): List of dictionary containing
            benchmark results.

    Returns:
        None.

    """
    repo = git.Repo(search_parent_directories=True)

    filename = get_report_filename(filename)
    this_script_path = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(os.path.join(this_script_path, "reports")):
        os.mkdir(os.path.join(this_script_path, "reports"))

    # check if the file already exists
    if os.path.exists(filename):
        with open(filename) as infile:
            data = json.load(infile)
            data[str(repo.head.commit)] = bresults
    else:
        data = {str(repo.head.commit): bresults}

    with open(filename, "w") as outfile:
        json.dump(data, outfile, indent=4)


def save_comparison_result(rev_this, rev_other, slowers, fasters, sames):
    """Function to save benchmark comparison result.

    Result saved in directory_of_this_script/reports/benchmark_comp.json.

    Args:
        rev_this (str): Name of current commit.
        rev_other (str): Name of commit to compare against.
        slowers (list of dict): List of dictionaries containing
            slower comparison results.
        fasters (list of dict): List of dictionaries containing
            faster comparison results.
        sames (list of dict): List of dictionaries containing
            same comparison results.

    Returns:
        None.

    """
    data = {"runtime": f"{rev_this} / {rev_other}"}
    data["slowers"] = slowers
    data["fasters"] = fasters
    data["sames"] = sames
    filename = get_report_filename("benchmark_comp.json")
    with open(filename, "w") as outfile:
        json.dump(data, outfile, indent=4)


def list_benchmark_modules():
    """Function to list all benchmark modules.

    Returns:
        list of str: List of all filenames in the directory
            containing this script with name benchmark_*.

    """
    import glob

    dir_path = os.path.dirname(__file__)
    modules = glob.glob(os.path.join(dir_path, "benchmark_*"))
    return [f[len(dir_path) + 1 : -3] for f in modules]


def main_run(args):
    """Function to run benchmarks.

    Run benchmark on all modules in the directory
    containing this script with name benchmark_*

    """
    results = []
    modules = list_benchmark_modules()
    for m_loop in modules:
        m = try_importing(m_loop)
        if m:
            try:
                r = m.run()
                results.append(r)
            except AttributeError:
                print(f"Something is wrong with {m}")

    save_benchmark_result(results, args.output)


def main_compare(args):
    """Function to compare benchmark results.

    Exits:
        1: If the runtime of any one result of rev_this is
            slower than that of the runtime of rev_other by
            more than the threshold ratio.

    Returns:
        None: If does not exit.

    """
    rt = args.rev_this
    ro = args.rev_other
    repo = git.Repo(search_parent_directories=True)
    rev_this = str(repo.commit(rt))
    rev_other = str(repo.commit(ro))

    filename = get_report_filename(args.filename)

    with open(filename) as infile:
        data = json.load(infile)

    rev_this_benchmark = data[rev_this]
    rev_other_benchmark = data[rev_other]

    # lists to store results
    slowers = []
    fasters = []
    sames = []

    # helper function to print and store results
    def compare_helper(_this_t, _other_t, _N, _thread):
        ratio = _other_t / _this_t
        info = {
            "name": this_res["name"],
            "params": this_res["params"],
            "N": _N,
            "ratio": ratio,
        }
        if _thread:
            info["threads"] = _thread
            print(
                f"Threads: {_thread!s}, N: {_N}, ratio: {ratio:0.2f}"
            )
        else:
            print(f"N: {_N}, ratio: {ratio:0.2f}")

        if ratio < 1:
            print(
                f"\t{rt:6.6} is {ratio:0.2f} times slower than {ro:6.6}"
            )
            slowers.append(info)
        if ratio > 1:
            print(
                f"\t{rt:6.6} is {ratio:0.2f} times faster than {ro:6.6}"
            )
            fasters.append(info)
        if ratio == 1:
            print(f"\t{rt:6.6} and {ro:6.6} have the same speed")
            sames.append(info)

    for this_res in rev_this_benchmark:
        for other_res in rev_other_benchmark:
            if (
                this_res["name"] == other_res["name"]
                and this_res["params"] == other_res["params"]
            ):
                print(benchmark_desc(this_res["name"], this_res["params"]))
                print(
                    "\nShowing runtime "
                    f"{ro:6.6} ({rev_other:6.6}) / "
                    f"{rt:6.6} ({rev_this:6.6})"
                )
                print()

                # compare size scaling behavior
                for N_loop in this_res["Ns"]:
                    N = str(N_loop)
                    this_t = this_res["size_scale"][N]
                    other_t = other_res["size_scale"][N]
                    compare_helper(this_t, other_t, N, None)

                # compare thread scaling behavior
                num_threads = len(this_res["thread_scale"]) - 1
                for i in range(1, num_threads + 1):
                    for j, N in enumerate(this_res["Ns"]):
                        this_t = this_res["thread_scale"][i][j]
                        other_t = other_res["thread_scale"][i][j]
                        compare_helper(this_t, other_t, N, i)

                print("\n ----------------")

    save_comparison_result(rt, ro, slowers, fasters, sames)

    # exit 1 if too slow
    threshold = 0.70
    for info in slowers:
        if info["ratio"] < threshold:
            desc = benchmark_desc(info["name"], info["params"])
            print(f"TOO SLOW (beyond threshold of {threshold})")
            print("\t" + desc)
            print("\t\tratio = {}".format(info["ratio"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_run = subparsers.add_parser(
        name="run",
        description="Execute performance tests in various categories for "
        "specific data space sizes (N).",
    )
    parser_run.add_argument(
        "-o",
        "--output",
        nargs="?",
        default="benchmark.json",
        help="Specify which collection file to store results "
        "to or '-' for None, "
        "default='benchmark.json'.",
    )
    parser_run.set_defaults(func=main_run)

    parser_report = subparsers.add_parser(
        name="report", description="Display results from previous runs."
    )
    parser_report.add_argument(
        "filename",
        default="benchmark.json",
        nargs="?",
        help="The collection that contains the benchmark data"
        "default='benchmark.json'.",
    )
    parser_report.set_defaults(func=main_report)

    parser_compare = subparsers.add_parser(
        name="compare",
        description="Compare performance between two "
        "git-revisions of this repository. "
        "For example, to compare the current revision "
        "(HEAD) with the "
        f"'main' branch revision, execute `{sys.argv[0]} compare "
        "main HEAD`. In this specific "
        "case one could omit both arguments, since 'main'"
        " and 'HEAD' are the two "
        "default arguments.",
    )
    parser_compare.add_argument(
        "rev_other",
        default="main",
        nargs="?",
        help="The git revision to compare against. "
        "Valid arguments are  for example "
        "a branch name, a tag, a specific commit id, "
        "or 'HEAD', defaults to 'main'.",
    )
    parser_compare.add_argument(
        "rev_this",
        default="HEAD",
        nargs="?",
        help="The git revision that is benchmarked. "
        "Valid arguments are  for example "
        "a branch name, a tag, a specific commit id, "
        "or 'HEAD', defaults to 'HEAD'.",
    )
    parser_compare.add_argument(
        "--filename",
        default="benchmark.json",
        nargs="?",
        help="The collection that contains the benchmark data"
        "default='benchmark.json'.",
    )
    parser_compare.add_argument(
        "-f",
        "--fail-above",
        type=float,
        help="Exit with error code in case that the runtime ratio of "
        "the worst tested category between this and the other revision "
        "is above this value.",
    )
    parser_compare.set_defaults(func=main_compare)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_usage()
        sys.exit(2)
    args.func(args)
