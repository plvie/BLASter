#!/usr/bin/python3
import argparse
import numpy as np
from math import log, prod, pi, exp
from multiprocessing import cpu_count
from threadpoolctl import threadpool_limits
from time import perf_counter_ns


def read_matrix(input_file, verbose=False):
    data = []
    if input_file is None:
        if verbose:
            print("Supply a matrix in fpLLL format:")

        data.append(input())
        while data[-1][-2] != ']':
            data.append(input())
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            data.append(f.readline()[:-1])
            while data[-1][-2] != ']':
                data.append(f.readline()[:-1])

    # Strip away starting '[' and ending ']'
    data[0] = data[0][1:]
    data[-1] = data[-1][:-1]

    for i in range(len(data)):
        data[i] = list(map(int, data[i][1:-1].split(' ')))

    # Parse the matrix now
    # Give the basis with column vectors.
    return np.transpose(np.array(data, dtype=np.float64))


def output_matrix(output_file, basis):
    # Assume that the basis is given with column vectors as input.
    # However, output them as row vectors.
    basis = np.transpose(basis)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('[')
        for i in range(len(basis)):
            f.write('[' + ' '.join(map(str, basis[i])) + ']\n')
        f.write(']')


def seysen_reduce(R, U):
    """
    Seysen reduce a matrix R, recursive style, and store the result in U.
    See: Algorithm 7 from [KEF21].
    [KEF21] P. Kircher, T. Espitau, P.-A. Fouque. Towards faster
    polynomial-time lattice reduction.
    :param R: an upper-triangular matrix (having row vectors).
    :param U: a unimodular transformation U such that RU is Seysen-Reduced.
    """
    n, m = len(R), (len(R) + 1) // 2
    # R11, R12, R22 = R[:m, :m], R[:m, m:], R[m:, m:]

    if n == 1:
        # Base case
        U[0, 0] = 1
    elif n == 2:
        # Make sure RU is size-reduced, i.e. |R00*X + R01| <= |R00|/2
        U[0, 0] = U[1, 1] = 1
        U[0, 1] = -round(R[0, 1] / R[0, 0])
    else:
        # TODO: n == 3?
        seysen_reduce(R[:m, :m], U[:m, :m])
        seysen_reduce(R[m:, m:], U[m:, m:])

        S11 = R[:m, :m] @ U[:m, :m]
        S12 = R[:m, m:] @ U[m:, m:]

        S11_inv = np.linalg.inv(S11)
        W = np.rint(S11_inv @ S12)
        # Now take the fractional part of the entries of W.
        U[:m, m:] = -U[:m, :m] @ W


def lagrange_reduce(R, delta=.99):
    """
    Tries to perform lagrange reduction, on all the even or odd indices.
    :param R: upper-triangular matrix
    :param nthreads: number of threads that we can use.
    :return: pair of:
        1) a transformation matrix U such that RU is Lagrange-reduced,
        2) a bool whether some reduction happened.
    """
    n = len(R)
    U = np.identity(n, dtype=np.float64)
    is_modified, skip = False, False

    for pos in range(0, n - 1):
        if skip:
            skip = False
            continue

        b0x = R[pos, pos]  # vector b0
        b1x, b1y = R[pos, pos + 1], R[pos + 1, pos + 1]  # vector b1

        if b1x * b1x + b1y * b1y < delta * (b0x * b0x):
            is_modified = skip = True

            # Reduce by making a swap and size-reducing b0 w.r.t. b1.
            q = round((b0x * b1x) / (b1x * b1x + b1y * b1y))
            # [b0', b1'] = [b1, b0 - q b1] = [b0, b1] U, with U=[[0,1],[1,q]]
            U[pos, pos] = 0
            U[pos + 1, pos] = U[pos, pos + 1] = 1
            U[pos + 1, pos + 1] = -q
    return U, is_modified


def seysen_lll(B, delta, measure_time=True):
    """
    :param B: a basis, consisting of *column vectors*.
    :return: transformation matrix U such that BU is LLL reduced.
    """
    n = len(B)
    U = np.identity(n, dtype=np.float64)
    U1 = np.zeros((n, n), dtype=np.float64)

    t_qr, t_seysen, t_lagrange, t_matmul = 0, 0, 0, 0

    is_modified = True
    while is_modified:
        t1 = perf_counter_ns()
        R = np.linalg.qr(B @ U, mode='r')
        t2 = perf_counter_ns()
        # R is upper-triangular
        seysen_reduce(R, U1)
        t3 = perf_counter_ns()
        U2, is_modified = lagrange_reduce(R @ U1, delta)
        t4 = perf_counter_ns()
        U = U @ (U1 @ U2)
        t5 = perf_counter_ns()

        if measure_time:
            t_qr += t2 - t1
            t_seysen += t3 - t2
            t_lagrange += t4 - t3
            t_matmul += t5 - t4

    if measure_time:
        print(f"Time QR factorization: {t_qr:18,d} ns\n"
              f"Time Seysen reduction: {t_seysen:18,d} ns\n"
              f"Time Lagrange reduct.: {t_lagrange:18,d} ns\n"
              f"Time Matrix Multipli.: {t_matmul:18,d} ns")

    return B @ U, U


def get_profile(B):
    return abs(np.linalg.qr(B, mode='r').diagonal())


def rhf(profile):
    n = len(profile)
    return prod((profile[0]/profile[i])**(1.0/n) for i in range(n))**(1.0 / n)


def __main__(args):
    B = read_matrix(args.input, args.verbose)

    if args.verbose:
        print("Matrix is read.")

    # Assumption: B is a q-ary lattice, with a q-vector at the end.
    n, q = len(B), B[-1][-1]
    log_slope = -log(args.delta - 0.25) / 2
    log_det = sum(log(x) for x in B.diagonal())
    lhs, rhs = n * log(q), log_slope * n * (n-1) / 2 + log_det
    if args.verbose:
        print(f'q-ary vector: e^{{{lhs:.3f}}}, allowed: q > e^{{{rhs:.3f}}}')
    assert lhs > rhs, 'q-ary vector could be part of an LLL reduced basis!'

    Bred, U = seysen_lll(B, args.delta, args.verbose)

    if not args.quiet:
        print('U: \n', U, sep="")

    print_mat = args.output is not None
    if print_mat and args.input is not None and args.output == args.input:
        print_mat = (input('WARNING: input & output files are same!\n Continue? (y/n)?') == 'y')
    if print_mat:
        # Output the reduced basis.
        output_matrix(args.output, Bred)
    elif not args.quiet:
        print('B: \n', Bred, sep="")

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
    assert 0.25 < args.delta and args.delta < 1.0, \
           'Invalid value given for delta!'

    np.set_printoptions(linewidth=275, suppress=True)
    with threadpool_limits(limits=args.cores):
        __main__(args)
