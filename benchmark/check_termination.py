#!/usr/bin/python3
import os
import subprocess

cmd_blaster = "../python3 ../src/app.py -q"
error_file = open("termination-errors.txt", "a", encoding='utf8')


def parse_time_usage(time_output):
    times = time_output.strip().split(" ")
    # parts = time_output.split("\n")[1:4]
    # times = [part.split("\t")[1] for part in parts]
    return {'real': float(times[0]), 'user': float(times[1]), 'sys': float(times[2])}


def run_command(cmd):
    t = 300 # 5 minutes

    try:
        cmd = f"/usr/bin/time -f \"%e %U %S\" {cmd}"
        result = subprocess.run(cmd, text=True, shell=True, capture_output=True, timeout=t)
    except subprocess.TimeoutExpired as e:
        print(f"Timeout expired for command \"{cmd}\"", file=error_file)
        if e.stderr is not None:
            print(e.stderr.decode("utf-8"), file=error_file)
        # TODO: kill process.
        return None
    if result.returncode != 0:
        print(f"Non zero return code encountered for command \"{cmd}\"", file=error_file)
        print(result.stderr, file=error_file)
        return None

    return parse_time_usage(result.stderr)


def gen_lattice(m, lgq, seed, path):
    n = m//2
    run_command(f"latticegen -randseed {seed} q {m} {n} {lgq} p > {path}")


def run_blaster(m, lgq, seed, path):
    logfile = f"../logs/termination/lll_{m}_{lgq}_{seed}.csv"
    return run_command(f"{cmd_blaster} -i {path} -l {logfile}")


def run_blaster_deeplll(m, lgq, seed, path, depth):
    logfile = f"../logs/termination/deeplll{depth}_{m}_{lgq}_{seed}.csv"
    return run_command(f"{cmd_blaster} -i {path} -l {logfile} -d{depth}")


def __main__():
    # Print CSV header
    print("m,lgq,seed,real,user,sys", flush=True)

    for seed in range(10):
        for lgm in range(4, 6):
        # for lgm in range(6, 11):
            m = 2**lgm
            lo_lgq, hi_lgq = 10, 64 - (lgm - 2) * (lgm - 1) // 2  # my bound
            for lgq in range(lo_lgq, hi_lgq + 1):
                path = f"../input/termination/{m}_{lgq}_{seed}"
                gen_lattice(m, lgq, seed, path)
                result = run_blaster(m, lgq, seed, path)
                if not result:
                    break
                print(f"{m:4d},{lgq:2d},{seed:1d},{result['real']:6.2f},{result['user']:6.2f},{result['sys']:6.2f}", flush=True)
            # print(f"dimension 2^{lgm} terminates correctly up to q = 2^{low_lgq}")


if __name__ == "__main__":
    __main__()
