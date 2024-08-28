#!/usr/bin/python3
"""
Script to perform lattice reduction, using Seysen Size-Reduction, but with the aim of outputting a
lattice with quality similar to what LLL achieves.
"""

import argparse
from math import ceil, exp, gamma, log, pi
from multiprocessing import cpu_count
from sys import stderr
from threadpoolctl import threadpool_limits

import numpy as np

from lattice_io import read_qary_lattice, write_lattice
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
    Return the profile of a basis, i.e. log ||b_i*|| for i=1, ..., n.
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
            '--LLL', '-L', type=int, default=64,
            help='Size of blocks on which to call LLL locally')

    # DeepLLL depth parameter
    parser.add_argument(
            '--depth', '-d', type=int, default=1,
            help='Maximum allowed depth for "deep insertions" in deepLLL. 1 if not desired.')

    args = parser.parse_args()
    assert 0.25 < args.delta and args.delta < 1.0, 'Invalid value for delta!'
    assert args.LLL >= 2, 'LLL block size must be at least 2!'

    B = read_qary_lattice(args.input)
    n = B.shape[1]

    assert np.count_nonzero(B[:, 0]) == 1
    q = sum(B[:, 0])

    if args.verbose:
        # Assume a RHF of ~1.02
        log_slope = log(1.02)  # -log(args.delta - 0.25)
        log_det = sum(get_profile(B))
        norm_b1 = exp(log_slope * (n-1) + log_det / n)
        cmp = "<" if norm_b1 < q else ">="
        print(f'E[∥b₁∥] ~ {norm_b1:.2f} {cmp} {int(q):d} ',
              f'(GH: λ₁ ~ {gh(n) * exp(log_det/n):.2f})',
              file=stderr)

    # We use multiple cores mainly for parallelizing running LLL on blocks, so limit the number of
    # cores to this. Matrix multiplication may use too many cores without any gain.
    args.cores = max(1, min(args.cores, ceil(n / args.LLL)))

    # Perform Seysen-LLL reduction on basis B
    with threadpool_limits(limits=1):
        U, B_red, prof = seysen_lll(B, args)

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
        print('\n', str(prof), sep="", file=stderr)

    # Print profile
    if args.profile:
        prof = get_profile(B_red)
        print('\nProfile: [' + ' '.join([f'{x:.2f}' for x in prof]) + ']', file=stderr)
        print(f'Root Hermite factor: {rhf(prof):.6f}, ∥b_1∥ = {exp(prof[0]):.3f}', file=stderr)

    # Assert that applying U on the basis B indeed gives the reduced basis B_red.
    assert (B @ U == B_red).all()


if __name__ == '__main__':
    np.set_printoptions(linewidth=1000, threshold=2147483647, suppress=True)
    __main__()
