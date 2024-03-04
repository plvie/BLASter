#!/usr/bin/python3
import argparse
import numpy as np
from multiprocessing import cpu_count, current_process, set_start_method, Process, Queue
from time import perf_counter_ns


def read_matrix(input_file):
    data = []
    if input_file is None:
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
    return np.array(data, dtype=np.int64)


def _seysen_reduce_subprocess(R, nthreads, queue):
    queue.put(seysen_reduce(R, nthreads))
    queue.close()


def seysen_reduce(R, nthreads=1):
    """
    Seysen reduction on a matrix, recursive style.

    See: Algorithm 7 from [KEF21].

    [KEF21] P. Kircher, T. Espitau, P.-A. Fouque. Towards faster polynomial-time lattice reduction.

    :pararm R: an upper-triangular matrix (having row vectors).
    :return: (RU, U) where U is a unimodular transformation U and RU is Seysen-Reduced.
    """
    n, m = len(R), (len(R) + 1) // 2
    R11, R12, R22 = R[:m, :m], R[:m, m:], R[m:, m:]

    if n == 1:
        # Base case
        # TODO: unroll case n == 2
        # TODO: unroll case n == 3 (?)
        return R, np.array([[1]], dtype=np.int64)

    S22, U22 = None, None
    if nthreads > 1:
        # Split work over current process and another
        half = nthreads // 2

        queue = Queue(maxsize=1)
        sp = Process(target=_seysen_reduce_subprocess, args=(R22, half, queue, ))
        sp.start()

        S11, U11 = seysen_reduce(R11, half)
        S22, U22 = queue.get()

        sp.join()
        sp.close()
        queue.close()
    else:
        S11, U11 = seysen_reduce(R11)
        S22, U22 = seysen_reduce(R22)

    S11_inv = np.linalg.inv(S11)
    S12 = R12 @ U22
    W = np.rint(S11_inv @ S12)
    # Now take the fractional part of the entries of W.
    U12 = -U11 @ W
    S12 -= S11 @ W

    # Assemble the transformation U.
    U = np.block([[U11, U12], [np.zeros((n-m, m)), U22]])
    S = np.block([[S11, S12], [np.zeros((n-m, m)), S22]])
    return S, U


def check_reducedness(submat):
    """
    :param submat: 2x2 matrix consisting of 2 basis vectors B, that get Lagrange reduced if needed.
    :return: 2x2 transformation matrix U, such that BU is Lagrange reduced.
    """
    b0x, b0y = submat[0, 0], submat[1, 0]  # vector b0
    b1x, b1y = submat[0, 1], submat[1, 1]  # vector b1

    # TODO: make this 0.99 a parameter `delta`.
    if 0.99 * (b0x * b0x + b0y * b0y) <= b1x * b1x + b1y * b1y:
        return np.array([[1, 0], [0, 1]])

    # Reduce by making a swap and size-reducing b0 w.r.t. b1.
    q = round((b0x * b1x + b0y * b1y) / (b1x * b1x + b1y * b1y))
    # [b0', b1'] = [b1, b0 - q b1] = [b0, b1] U, where U equals:
    # [ 0  1 ]
    # [ 1 -q ]
    return np.array([[0, 1], [1, -q]])


def half_lagrange_reduce(R, nthreads=1, even=True):
    """
    Tries to perform lagrange reduction, on all the even or odd indices.
    :param R: upper-triangular matrix
    :param nthreads: number of threads that we can use.
    :param even: when true, reduces [b0, b1], [b2, b3], ..., otherwise [b1, b2], [b3, b4], ...
    :return: transformation matrix U to lagrange such that RU is half-lagrange-reduced.
    """
    n = len(R)
    positions = range(0 if even else 1, n - 1, 2)
    submatrices = [R[p:p+2, p:p+2] for p in positions]

    U = np.identity(n)
    # with Pool(nthreads) as pool:
    #     Us = pool.map(check_reducedness, submatrices, n // (2*nthreads))
    Us = list(map(check_reducedness, submatrices))
    for i, p in enumerate(positions):
        U[p:p+2, p:p+2] = Us[i]
    return U


def seysen_lll(B, nthreads, measure_time=True):
    """
    :param B: a basis, consisting of *column vectors*.
    :return: transformation matrix U such that BU is LLL reduced.
    """
    n = len(B)
    U = np.identity(n)
    seysen_cores = 1 #nthreads

    time_qr, time_seysen, time_lagrange, time_matmul = 0, 0, 0, 0

    is_modified = True
    while is_modified:
        t1 = perf_counter_ns()

        R1 = np.linalg.qr(B, mode='r')

        t2 = perf_counter_ns()
        time_qr += t2 - t1
        t1 = t2

        # R1 is upper-triangular
        print("Seysen #1")
        S1, U1 = seysen_reduce(R1, seysen_cores)

        t2 = perf_counter_ns()
        time_seysen += t2 - t1
        t1 = t2

        U2 = half_lagrange_reduce(S1, nthreads, True)

        t2 = perf_counter_ns()
        time_lagrange += t2 - t1
        t1 = t2

        is_modified = (U2 != np.identity(n)).any()
        U12 = U1 @ U2
        U = U @ U12
        B = B @ U12

        t2 = perf_counter_ns()
        time_matmul += t2 - t1
        t1 = t2

        R2 = np.linalg.qr(B, mode='r')

        t2 = perf_counter_ns()
        time_qr += t2 - t1
        t1 = t2

        # R2 is upper-triangular
        print("Seysen #2")
        S2, U3 = seysen_reduce(R2, seysen_cores)

        t2 = perf_counter_ns()
        time_seysen += t2 - t1
        t1 = t2

        U4 = half_lagrange_reduce(S2, nthreads, False)

        t2 = perf_counter_ns()
        time_lagrange += t2 - t1
        t1 = t2

        is_modified = is_modified or (U4 != np.identity(n)).any()
        U34 = U3 @ U4
        U = U @ U34
        B = B @ U34

        t2 = perf_counter_ns()
        time_matmul += t2 - t1
        t1 = t2

    print(f"Time QR factorization: {time_qr:18,d} ns\n"
          f"Time Seysen reduction: {time_seysen:18,d} ns\n"
          f"Time Lagrange reduct.: {time_lagrange:18,d} ns\n"
          f"Time Matrix Multipli.: {time_matmul:18,d} ns")

    return B, U


def __main__(args):
    B = np.transpose(read_matrix(args.i))
    Bp, U = seysen_lll(B, args.cores)
    print("U: \n", U, sep="")
    print("B: \n", Bp, sep="")


###############################################################################
if __name__ == '__main__':
    set_start_method('spawn')
    # Parse the command line arguments:
    parser = argparse.ArgumentParser(
            prog='SeysenLLL',
            description='LLL-reduce a lattice using seysen reduction',
            epilog='Input/output is formatted as is done in fpLLL')
    parser.add_argument(
            '--cores', type=int, default=cpu_count() // 2,
            help='number of cores to be used')
    # parser.add_argument(
    #         '--delta', type=float, default=0.25,
    #         help='delta factor for Lovasz condition')

    parser.add_argument('-i', type=str, help='Input file (default=stdin)')
    parser.add_argument('-o', type=str, help='Output file (default=stdout)')
    args = parser.parse_args()

    np.set_printoptions(linewidth=275)
    __main__(args)
