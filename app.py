#!/usr/bin/python3
"""
Script to perform lattice reduction, using Seysen Size-Reduction, but with the aim of outputting a
lattice with quality similar to what LLL achieves.
"""

import argparse
from math import exp, log, prod, sqrt
from multiprocessing import cpu_count
from time import perf_counter_ns

import numpy as np
from threadpoolctl import threadpool_limits

from lattice_io import read_matrix, write_matrix
from seysen import seysen_lll


def get_profile(B):
    """
    Returns the (exp-)profile of a basis, i.e. ||b_i*|| for i=1, ..., n.
    Note: some people define the profileas log ||b_i*|| instead!
    :param B: basis for a lattice
    """
    return abs(np.linalg.qr(B, mode='r').diagonal())


def rhf(profile):
    """
    Return the root Hermite factor, given the profile of some basis.
    :param profile: profile belonging to some basis of some lattice
    """
    n = len(profile)
    return prod((profile[0]/profile[i])**(1.0/n) for i in range(n))**(1.0 / n)


def __main__(args):
    B = read_matrix(args.input, args.verbose)
    n = len(B)

    # Assumption: B is a q-ary lattice.
    q = B[-1][-1]

    # Assume a RHF of ~1.02 
    log_slope = log(1.02)  # -log(args.delta - 0.25) / 2
    log_det = sum(log(x) for x in B.diagonal())
    expected_shortest = exp(log_slope * (n-1) / 2 + log_det / n)
    if args.verbose:
        print(f'Expected shortest vector: {expected_shortest:.3f} <(?) {int(q):d}')
    assert expected_shortest < q, 'q-ary vector could be part of an LLL reduced basis!'

    # Perform Seysen-LLL reduction
    U, prof = seysen_lll(B, args.delta)

    # Print U
    if not args.quiet:
        print('U: \n', U, sep="")

    # Print B @ U
    print_mat = args.output is not None
    if print_mat and args.input is not None and args.output == args.input:
        print_mat = (input('WARNING: input & output files are same!\n Continue? (y/n)?') == 'y')
    Bred = B @ U
    if print_mat:
        # Output the reduced basis.
        write_matrix(args.output, Bred)
    elif not args.quiet:
        print('B: \n', Bred, sep="")

    # Print time consumption
    if args.verbose:
        print(str(prof))

    # Print profile
    if args.profile:
        prof = get_profile(Bred)
        print('Profile: [', ' '.join(['{:.2f}'.format(x) for x in prof]),
              ']', sep='')
        print(f'Root hermite factor: {rhf(prof):.6f}')


###############################################################################
if __name__ == '__main__':
    # Parse the command line arguments:
    parser = argparse.ArgumentParser(
            prog='SeysenLLL',
            description='LLL-reduce a lattice using seysen reduction',
            epilog='Input/output is formatted as is done in fpLLL')

    parser.add_argument(
            '--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument(
            '--cores', type=int, default=cpu_count() // 2,
            help='number of cores to be used')
    parser.add_argument(
            '--quiet', '-q', action='store_true',
            help='Quiet mode will not output the output basis')
    parser.add_argument(
            '--profile', '-p', action='store_true',
            help='Give information on the profile of the output basis')
    parser.add_argument(
            '--delta', type=float, default=0.99,
            help='delta factor for Lovasz condition')
    parser.add_argument(
            '--input', '-i', type=str,
            help='Input file (default=stdin)')
    parser.add_argument(
            '--output', '-o', type=str,
            help='Output file (default=stdout)')

    args = parser.parse_args()
    assert 0.25 < args.delta and args.delta < 1.0, 'Invalid value given for delta!'

    np.set_printoptions(linewidth=200, threshold=2147483647, suppress=True)
    with threadpool_limits(limits=args.cores):
        __main__(args)
