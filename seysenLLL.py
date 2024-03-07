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

    # Simple code: assume the matrix is a q-ary lattice, with a q-vector at the
    # end.
    # Now check if GH < q.
    n, q = len(data), data[-1][-1]
    GH = (n / (2 * pi * exp(1)))**.5
    GH *= exp(sum(log(data[i][i]) for i in range(n)) / n)
    if args.verbose:
        print(f'q-ary vector is at {q/GH:.6f}*GH.')
    assert GH < q

    # Parse the matrix now
    # Give the basis with column vectors.
    return np.transpose(np.array(data, dtype=np.int64))


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


def half_lagrange_reduce(R, do_even, delta=.99):
    """
    Tries to perform lagrange reduction, on all the even or odd indices.
    :param R: upper-triangular matrix
    :param nthreads: number of threads that we can use.
    :param do_even: when true, reduces [b0, b1], [b2, b3], ..., otherwise [b1, b2], [b3, b4], ...
    :return: transformation matrix U to lagrange such that RU is half-lagrange-reduced.
    """
    n = len(R)
    U = np.identity(n, dtype=np.int64)
    is_modified = False

    for pos in range(0 if do_even else 1, n - 1, 2):
        b0x, b0y = R[pos, pos], R[pos + 1, pos]  # vector b0
        b1x, b1y = R[pos, pos + 1], R[pos + 1, pos + 1]  # vector b1

        # TODO: make this 0.99 a parameter `delta`.
        if b1x * b1x + b1y * b1y < delta * (b0x * b0x + b0y * b0y):
            is_modified = True

            # Reduce by making a swap and size-reducing b0 w.r.t. b1.
            q = round((b0x * b1x + b0y * b1y) / (b1x * b1x + b1y * b1y))
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
    U = np.identity(n, dtype=np.int64)
    U1 = np.zeros((n, n), dtype=np.int64)
    U3 = np.zeros((n, n), dtype=np.int64)

    time_qr, time_seysen, time_lagrange, time_matmul = 0, 0, 0, 0

    is_modified = True
    while is_modified:
        t1 = perf_counter_ns()

        R1 = np.linalg.qr(B @ U, mode='r')
        # R1 = np.linalg.qr(B, mode='r')

        t2 = perf_counter_ns()
        time_qr += t2 - t1
        t1 = t2

        # R1 is upper-triangular
        seysen_reduce(R1, U1)
        R1 = R1 @ U1

        t2 = perf_counter_ns()
        time_seysen += t2 - t1
        t1 = t2

        U2, is_modified = half_lagrange_reduce(R1, True, delta)
        # is_modified = (U2 != np.identity(n)).any()

        t2 = perf_counter_ns()
        time_lagrange += t2 - t1
        t1 = t2

        # U12 = U1 @ U2
        # U = U @ U12
        # B = B @ U12

        # t2 = perf_counter_ns()
        # time_matmul += t2 - t1
        # t1 = t2

        R2 = np.linalg.qr(R1 @ U2, mode='r')
        # R2 = np.linalg.qr(B, mode='r')

        t2 = perf_counter_ns()
        time_qr += t2 - t1
        t1 = t2

        # R2 is upper-triangular
        seysen_reduce(R2, U3)
        R2 = R2 @ U3

        t2 = perf_counter_ns()
        time_seysen += t2 - t1
        t1 = t2

        U4, is_modified2 = half_lagrange_reduce(R2, False, delta)
        is_modified = is_modified or is_modified2

        t2 = perf_counter_ns()
        time_lagrange += t2 - t1
        t1 = t2

        # U34 = U3 @ U4
        # U = U @ U34
        # B = B @ U34
        U = U @ U1 @ U2 @ U3 @ U4

        t2 = perf_counter_ns()
        time_matmul += t2 - t1
        t1 = t2

    if measure_time:
        print(f"Time QR factorization: {time_qr:18,d} ns\n"
              f"Time Seysen reduction: {time_seysen:18,d} ns\n"
              f"Time Lagrange reduct.: {time_lagrange:18,d} ns\n"
              f"Time Matrix Multipli.: {time_matmul:18,d} ns")

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

    with threadpool_limits(limits=args.cores):
        __main__(args)
