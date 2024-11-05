#!/usr/bin/python3
import os
import subprocess
import sys

from flatter_conversion import convert_logfiles


# Specify which lattices we want to test:
mqs = [
    (128, 631),  # latticegen q 2 1 10 p
    (256, 829561),  # latticegen q 2 1 20 p
    (512, 968665207),  # latticegen q 2 1 30 p
    (1024, 829561),  # latticegen q 2 1 20 p
]

seeds = range(10)
lattices = [(m, q, seed, f"../input/{m}_{q}_{seed}") for (m, q) in mqs for seed in seeds]


def run_command(cmd):
    print(f"Executing \"{cmd}\".", flush=True)
    return subprocess.run(cmd, shell=True)


def gen_lattice(m, q, seed, path):
    n = m//2
    result = run_command(f"latticegen -randseed {seed} q {m} {n} {q} q > {path}")
    if result.returncode != 0:
        print(result.stderr)
        exit(1)


def run_seysenlll(m, q, seed, path):
    logfile = f"logs/lll_{m}_{q}_{seed}.csv"
    result = run_command(f"../src/app.py -pqvi {path} --logfile {logfile}")
    if result.returncode != 0:
        print(result.stderr)
        os.remove(logfile)


def run_seysendeeplll(m, q, seed, path, depth):
    logfile = f"logs/deeplll_d{depth}_{m}_{q}_{seed}.csv"
    result = run_command(f"../src/app.py -pqvi {path} --depth {depth} --logfile {logfile}")
    if result.returncode != 0:
        print(result.stderr)
        os.remove(logfile)


def run_flatter(m, q, seed, path, num_threads):
    flogfile = f"flatter_logs/{m}_{q}_{seed}.log"
    cmd = f"OMP_NUM_THREADS={num_threads} FLATTER_LOG={flogfile} ~/.local/bin/flatter -q {path}"
    result = run_command(cmd)
    if result.returncode != 0:
        print(result.stderr)
        exit(1)

    plogfile = f"logs/flatter_{m}_{q}_{seed}.csv"
    convert_logfiles(flogfile, plogfile)


def extract_times(time_output):
    """
    Extract real time, user time and system time from stdout,
    when timing a command with `time [...]`.
    """
    lines = time_output.split('\n')
    return lines[1][5:-1], lines[2][5:-1], lines[3][4:-1]


def benchmark(cmd):
    result = subprocess.run(cmd, capture_output=True, shell=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        exit(1)
    return ', '.join(extract_times(result.stderr))


def __main__():
    has_cmd = 'lattices' in sys.argv[1:]
    if has_cmd:
        for lat in lattices:
            gen_lattice(*lat)

    for i, arg in enumerate(sys.argv[1:]):
        is_cmd = True
        if arg == 'dim':
            assert 2 + i < len(sys.argv), "dim param expected!"
            dim = int(sys.argv[2 + i])
            assert dim in [m for (m, q) in mqs], "Unknown dimension"
            curq = [q for (m, q) in mqs if m == dim]
            assert len(curq) == 1
            curq = curq[0]
            lattices = [(dim, curq, seed, f"../input/{dim}_{curq}_{seed}") for seed in seeds]

        elif arg == 'seysen':
            for lat in lattices:
                run_seysenlll(*lat)
        elif arg == 'depth':
            assert 2 + i < len(sys.argv), "depth param expected!"
            depth = int(sys.argv[2 + i])
            for lat in lattices:
                run_seysendeeplll(*lat, depth)
        elif arg == 'flatter':
            assert 2 + i < len(sys.argv), "num_threads param expected!"
            num_threads = int(sys.argv[2 + i])
            for lat in lattices:
                run_flatter(*lat, num_threads)
        else:
            is_cmd = False
        has_cmd = has_cmd or is_cmd

    if not has_cmd:
        print(f"Usage: {sys.argv[0]} [dim d|lattices|seysen|depth d|flatter num_threads]")


if __name__ == "__main__":
    __main__()
