#!/usr/bin/python3
"""
Script to perform lattice reduction, using Seysen Size-Reduction, but with the aim of outputting a
lattice with quality similar to what LLL achieves.
"""

import argparse
from math import exp, gamma, log, pi, prod
from multiprocessing import cpu_count
from sys import stderr
from threadpoolctl import threadpool_limits

import numpy as np

from lattice_io import read_matrix, write_matrix
from seysen import seysen_lll


def gh(dim):
    """
    Return the Gaussian Heuristic at dimension n. This gives a prediction of
    the length of the shortest vector in a lattice of unit volume.
    :param n: lattice dimension
    :returns: GH(n)
    """
    if dim >= 100:
        return float(dim / (2*pi*exp(1)))**0.5
    return float(gamma(1.0 + 0.5 * dim)**(1.0 / dim) / pi**0.5)


def get_profile(B):
    """
    Returns the profile of a basis, i.e. log ||b_i*|| for i=1, ..., n.
    :param B: basis for a lattice
    """
    return [log(abs(d_i)) for d_i in np.linalg.qr(B, mode='r').diagonal()]


def rhf(profile):
    """
    Return the root Hermite factor, given the profile of some basis, i.e.
        rhf(B) = (||b_0|| / det(B)^{1/n})^{1/(n-1)}.
    :param profile: profile belonging to some basis of some lattice
    """
    n = len(profile)
    return exp((profile[0] - sum(profile) / n) / (n - 1))


def __main__():
    # Parse the command line arguments:
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

    # Output profile?
    parser.add_argument(
            '--profile', '-p', action='store_true',
            help='Give information on the profile of the output basis')
    # Do not output reduced basis?
    parser.add_argument(
            '--quiet', '-q', action='store_true',
            help='Quiet mode will not output the output basis')

    # Lovasz condition
    parser.add_argument(
            '--delta', type=float, default=0.99,
            help='delta factor for Lovasz condition')

    # LLL block size
    parser.add_argument(
            '--LLL', '-L', type=int, default=1,
            help='Size of blocks on which to call LLL locally')

    args = parser.parse_args()
    assert 0.25 < args.delta and args.delta < 1.0, 'Invalid value given for delta!'

    if args.LLL == 1 and args.verbose:
        print('Note: LLL block size is <=2. '
              'Tip: Add `--LLL <blocksize>` to run LLL locally, '
              'which usually provides a speed up.', file=stderr)

    B = np.ascontiguousarray(read_matrix(args.input, args.verbose))
    n = len(B)

    # Assumption: B is a q-ary lattice.
    q = B[-1, 0] if all(B[:-1, 0] == 0) else B[-1, -1]

    # Assume a RHF of ~1.02
    log_slope = log(1.02)  # -log(args.delta - 0.25)
    log_det = sum(get_profile(B))
    expected_shortest = exp(log_slope * (n-1) + log_det / n)
    if args.verbose:
        cmp = "<" if expected_shortest < q else ">="
        print(f'E[∥b₁∥] ~ {expected_shortest:.2f} {cmp} {int(q):d} ',
              f'(GH: λ₁ ~ {gh(n) * exp(log_det/n):.2f})',
              file=stderr)

    # Perform Seysen-LLL reduction on basis B
    with threadpool_limits(limits=1):
        U, B_red, prof = seysen_lll(B, args)

    # Write B_red to the output file
    print_mat = args.output is not None
    if print_mat and args.input is not None and args.output == args.input:
        print_mat = input('WARNING: input & output files are same!\n Continue? (y/n)?') == 'y'
    if print_mat:
        write_matrix(args.output, B_red)
    elif not args.quiet:
        print(B_red.astype(np.int64))

    # Print time consumption
    if args.verbose:
        print('\n', str(prof), sep="", file=stderr)

    # Print profile
    if args.profile:
        prof = get_profile(B_red)
        print('\nProfile: [' + ' '.join([f'{x:.2f}' for x in prof]) + ']',
              f'Root Hermite factor: {rhf(prof):.6f}, ||b_1|| = {exp(prof[0]):.3f}', sep='\n', file=stderr)

    assert (B @ U == B_red).all()

###############################################################################
if __name__ == '__main__':
    np.set_printoptions(linewidth=1000, threshold=2147483647, suppress=True)
    __main__()
