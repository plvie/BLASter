#!/usr/bin/python3
import os
import subprocess
import sys

from flatter_conversion import convert_logfiles


# Specify which lattices we want to test:
mlgqs = [(128, 10), (256, 20)]  #, (512, 36)]
# mlgqs = [(512, 32)]

seeds = range(10)
lattices = [(m, lgq, seed, f"../input/{m}_{lgq}_{seed}") for (m, lgq) in mlgqs for seed in seeds]


def run_command(cmd):
    print(f"Executing \"{cmd}\".", flush=True)
    return subprocess.run(cmd, shell=True)


def gen_lattice(m, lgq, seed, path):
    n = m//2
    result = run_command(f"latticegen -randseed {seed} q {m} {n} {lgq} p > {path}")
    if result.returncode != 0:
        print(result.stderr)
        exit(1)


def run_seysenlll(m, lgq, seed, path):
    logfile = f"logs/lll_{m}_{lgq}_{seed}.csv"
    result = run_command(f"../src/app.py -pqvi {path} --logfile {logfile}")
    if result.returncode != 0:
        print(result.stderr)
        os.remove(logfile)
        # exit(1)


def run_seysendeeplll(m, lgq, seed, path, depth):
    logfile = f"logs/deeplll_d{depth}_{m}_{lgq}_{seed}.csv"
    result = run_command(f"../src/app.py -pqvi {path} --depth {depth} --logfile {logfile}")
    if result.returncode != 0:
        print(result.stderr)
        os.remove(logfile)
        # exit(1)


def run_flatter(m, lgq, seed, path):
    flogfile = f"flatter_logs/{m}_{lgq}_{seed}.log"
    result = run_command(f"FLATTER_LOG={flogfile} ~/.local/bin/flatter -q {path}")
    if result.returncode != 0:
        print(result.stderr)
        exit(1)
    plogfile = f"logs/flatter_{m}_{lgq}_{seed}.csv"
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
    if '--lattices' in sys.argv[1:]:
        for lat in lattices:
            gen_lattice(*lat)

    if '--seysen' in sys.argv[1:]:
        for lat in lattices:
            run_seysenlll(*lat)

    if '--depth' in sys.argv[1:]:
        depth = int(sys.argv[2 + sys.argv[1:].index('--depth')])
        for lat in lattices:
            run_seysendeeplll(*lat, depth)

    if '--flatter' in sys.argv[1:]:
        for lat in lattices:
            run_flatter(*lat)


if __name__ == "__main__":
    __main__()
#    print('   m,    n,  q,    method, real, user, sys')
#    for m in range(16, 1024+1, 16):
#        n = m // 2
#        q = 32 # round(m / 12 + 5)
#
#        # Run SeysenLLL
#        name, cmd = 'SeysenLLL', f'time {{ latticegen q {m} {n} {q} p | ../src/app.py -q;}}'
#        print(f'{m:4}, {n:4}, {q:2}, {name}, ', flush=True, end='')
#        print(benchmark(cmd), flush=True)
#        # # Run Flatter
#        # name, cmd = '  Flatter', f'~/.local/bin/flatter -q {file}'
#        # print(f'{m:4}, {n:4}, {q:2}, {name}, ', end='')
#        # print(benchmark(cmd), end='', flush=True)
#        # # Run FPLLL
#        # name, cmd = '    fpLLL', f'fplll {file} > /dev/null'
#        # print(f'{m:4}, {n:4}, {q:2}, {name}, ', end='')
#        # print(benchmark(cmd), end='', flush=True)
