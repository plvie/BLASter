#!/usr/bin/env python3
"""
Script to perform lattice reduction, using Seysen Size-Reduction, but with the aim of outputting a
lattice with quality similar to what LLL achieves.
"""

import argparse
from multiprocessing import cpu_count
from sys import stderr
from threadpoolctl import threadpool_limits
from math import log2, ceil

import numpy as np

from lattice_io import read_qary_lattice, write_lattice
from seysen import seysen_lll
from stats import gaussian_heuristic, rhf, slope, get_profile


def __main__():
    parser = argparse.ArgumentParser(
            prog='SeysenLLL',
            description='LLL-reduce a lattice using seysen reduction',
            epilog='Input/output is formatted as is done in fpLLL')

    # Global settings
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument(
            '--cores', '-j', type=int, default=cpu_count() // 2,
            help='number of cores to be used')

    # I/O arguments
    parser.add_argument('--input', '-i', type=str, help='Input file (default=stdin)')
    parser.add_argument('--output', '-o', type=str, help='Output file (default=stdout)')
    parser.add_argument('--logfile', '-l', type=str, default=None, help='Logging file')
    parser.add_argument(
            '--profile', '-p', action='store_true',
            help='Give information on the profile of the output basis')
    parser.add_argument(
            '--quiet', '-q', action='store_true',
            help='Quiet mode will not output the output basis')
    parser.add_argument(
            '--anim', '-a', type=str,
            help='Output a gif-file animating the basis profile during lattice reduction')

    # LLL parameters
    parser.add_argument(
            '--delta', type=float, default=0.99,
            help='delta factor for Lovasz condition')
    parser.add_argument(
            '--LLL', '-L', type=int, default=64,
            help='Size of blocks on which to call LLL/DeepLLL/BKZ locally & in parallel')

    # Parameters specific to DeepLLL:
    parser.add_argument(
            '--depth', '-d', type=int, default=0,
            help='Maximum allowed depth for "deep insertions" in deepLLL. 0 if not desired.')

    # Parameters specific to BKZ:
    parser.add_argument(
            '--beta', '-b', type=int, default=0,
            help='Blocksize used within BKZ. 0 if not desired.')
    parser.add_argument(
            '--max_tours', '-t', type=int, default=0,
            help='Maximum number of tours allowed to perform. 0 if unlimited.')

    # Parse the command line arguments
    args = parser.parse_args()

    # Perform sanity checks
    assert 0.25 < args.delta and args.delta < 1.0, 'Invalid value for delta!'
    assert args.LLL >= 2, 'LLL block size must be at least 2!'

    assert not args.depth or not args.beta, 'Cannot run combination of DeepLLL and BKZ!'
    if args.beta:
        assert 2 * args.beta <= args.LLL, 'LLL blocksize is not large enough for BKZ!'

    B = read_qary_lattice(args.input)
    n = B.shape[1]

    if args.verbose:
        # Assume a RHF of ~1.02
        log_slope = log2(1.02)  # -log2(args.delta - 0.25)
        log_det = sum(get_profile(B))
        norm_b1 = 2.0**(log_slope * (n-1) + log_det / n)

        comparison = ""
        if np.count_nonzero(B[:, 0]) == 1:
            q = sum(B[:, 0])
            cmp = "<" if norm_b1 < q else ">="
            comparison = f'{cmp} {int(q):d} '
        print(f'E[∥b₁∥] ~ {norm_b1:.2f} {comparison}'
              f'(GH: λ₁ ~ {gaussian_heuristic(B):.2f})',
              file=stderr)

    # We use multiple cores mainly for parallelizing running LLL on blocks, so limit the number of
    # cores to this. Matrix multiplication may use too many cores without any gain.
    args.cores = max(1, min(args.cores, ceil(n / args.LLL)))

    # Perform Seysen-LLL reduction on basis B
    U, B_red, tprof = seysen_lll(B, args)

    # Write B_red to the output file
    print_mat = args.output is not None
    if print_mat and args.output == args.input:
        print_mat = input('WARNING: input & output files are same!\nContinue? (y/n) ') == 'y'
    if print_mat:
        write_lattice(args.output, B_red)
    elif not args.quiet:
        print(B_red.astype(np.int64))

    # Print time consumption
    if args.verbose:
        print('\n', str(tprof), sep="", file=stderr)

    # Print basis profile
    if args.profile:
        prof = get_profile(B_red)
        print('\nProfile: [' + ' '.join([f'{x:.2f}' for x in prof]) + ']', file=stderr)
        print(f'Root Hermite factor: {rhf(prof):.6f}, ∥b_1∥ = {2.0**(prof[0]):.3f}', file=stderr)
        print(f'Profile avg slope: {slope(prof):.6f}', file=stderr)

    # Assert that applying U on the basis B indeed gives the reduced basis B_red.
    assert (B @ U == B_red).all()


if __name__ == '__main__':
    np.set_printoptions(linewidth=1000, threshold=2147483647, suppress=True)
    with threadpool_limits(limits=1):
        __main__()
