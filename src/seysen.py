"""
LLL reduction with Seysen instead of size reduction.
"""

from math import sqrt
from multiprocessing import Pool
from sys import stderr
from time import perf_counter_ns
import numpy as np

from seysen_lll import perform_lll_on_blocks

has_eigen = True
try:
    from eigenpy import LLT
except ImportError:
    print('Library `Eigen3` is not found.')
    has_eigen = False

# Global toggle whether to use Block_cholesky from [KEF21].
# If set to False, we will use the function 'linalg.qr' from numpy.
USE_BLOCK_CHOLESKY = False


class TimeProfile:
    """
    Object containing time spent on different parts within Seysen-LLL reduction.
    """

    def __init__(self):
        self.num_iterations = 0
        self.time_qr = self.time_lll = self.time_seysen = self.time_matmul = 0

    @classmethod
    def iteration(cls, t1, t2, t3, t4, t5):
        prof = cls()
        prof.num_iterations = 1
        prof.time_qr = t2 - t1
        prof.time_lll = t3 - t2
        prof.time_seysen = t4 - t3
        prof.time_matmul = t5 - t4
        return prof

    def __iadd__(self, other):
        self.num_iterations += other.num_iterations
        self.time_qr += other.time_qr
        self.time_lll += other.time_lll
        self.time_seysen += other.time_seysen
        self.time_matmul += other.time_matmul
        return self

    def __str__(self):
        return (f"Iterations: {self.num_iterations}\n"
                f"Time QR factorization: {self.time_qr:18,d} ns\n"
                f"Time LLL    reduction: {self.time_lll:18,d} ns\n"
                f"Time Seysen reduction: {self.time_seysen:18,d} ns\n"
                f"Time Matrix Multipli.: {self.time_matmul:18,d} ns")


# Turns out, this is slower than `R = np.linalg.qr(B, 'r')`.
def block_cholesky(G, R):
    """
    Return the Cholesky decomposition of G using a recursive strategy, based on [KEF21], but using
    2 instead of 4 matrix multiplications.
    [KEF21] Towards Faster Polynomial-Time Lattice Reduction
    :param G: gram matrix of some basis.
    :param R: result R will be upper triangular satisfying: `R^T R = G`.
    :return: None! Result is stored in R.
    """
    if len(G) == 1:
        R[0, 0] = np.array([[sqrt(G[0, 0])]])
    else:
        m = len(G) // 2
        # Given G = B^T B, find upper-triangular R = [[R0, S], [0, R1]],
        # such that R^T R = G.

        # Line 3: Recover R0.
        block_cholesky(G[:m, :m], R[:m, :m])
        # Line 4: S satisfies: G01 = R0^T S --> (S = R0^{-T} G01)
        R[:m, m:] = np.linalg.inv(R[:m, :m]).transpose() @ G[:m, m:]
        # Line 5: Recover R1 = BlockCholesky(G11 - S^T S)

        schur = G[m:, m:] - R[:m, m:].transpose() @ R[:m, m:]
        block_cholesky(schur, R[m:, m:])


def eigen_cholesky(G):
    return LLT(G).matrixL().transpose()


def qr_decompose(B):
    if USE_BLOCK_CHOLESKY:
        if has_eigen:
            # return eigen_cholesky(Bf.transpose() @ Bf)
            return eigen_cholesky(B.transpose() @ B)
        else:
            Bf = B.astype(np.float64)
            R = np.identity(len(B), dtype=np.float64)
            block_cholesky(Bf.transpose() @ Bf, R)
            return R
    else:
        R = np.linalg.qr(B, mode='r')
        for i in range(len(B)):
            if R[i, i] < 0:
                # Negate this row.
                R[i, i:] *= -1

        return R


def seysen_reduce(R, U):
    """
    Seysen reduce a matrix R, recursive style, and store the result in U.
    See: Algorithm 7 from [KEF21].
    [KEF21] P. Kircher, T. Espitau, P.-A. Fouque. Towards faster
    polynomial-time lattice reduction.
    :param R: an upper-triangular matrix (having row vectors).
    :param U: a unimodular transformation U such that RU is Seysen-Reduced.
    :return: None! The result is stored in U.
    """
    n, m = len(R), len(R) // 2
    # TODO: Write an iterative version that beats the recursive version.

    if n == 1:
        # Base case
        U[0, 0] = 1
    elif n == 2:
        # Make sure RU is size-reduced, i.e. |R00*X + R01| <= |R00|/2
        U[0, 0] = U[1, 1] = 1
        U[0, 1] = -round(R[0, 1] / R[0, 0])
    else:
        # TODO: Unroll loop for n == 3?
        # R11, R12, R22 = R[:m, :m], R[:m, m:], R[m:, m:]

        seysen_reduce(R[:m, :m], U[:m, :m])
        seysen_reduce(R[m:, m:], U[m:, m:])

        S11 = R[:m, :m] @ U[:m, :m]
        S12 = R[:m, m:] @ U[m:, m:]

        W = np.rint(np.linalg.inv(S11) @ S12)  # W = round(S11^{-1} S12).
        # Now take the fractional part of the entries of W.
        U[:m, m:] = -U[:m, :m] @ W


def lagrange_reduce(R, delta=.99):
    """
    Tries to perform lagrange reduction, on all the even or odd indices.
    :param R: upper-triangular matrix
    :param delta: delta-factor used in the Lovasz condition
    :return: pair of:
        1) a transformation matrix U such that RU is Lagrange-reduced,
        2) a bool whether some reduction happened.
    """
    n = len(R)
    U = np.identity(n, dtype=np.float64)
    last_change = -2

    for pos in range(0, n - 1):
        if last_change == pos - 1:
            # The disjoint lagrange reductions yield independent transformation matrices.
            continue

        b0x = R[pos, pos]  # vector b0
        b1x, b1y = R[pos, pos + 1], R[pos + 1, pos + 1]  # vector b1

        if b1x * b1x + b1y * b1y < delta * (b0x * b0x):
            last_change = pos

            # Reduce by making a swap and size-reducing b0 w.r.t. b1.
            q = round((b0x * b1x) / (b1x * b1x + b1y * b1y))
            # [b0', b1'] = [b1, b0 - q b1] = [b0, b1] U, with U=[[0,1],[1,q]]
            U[pos, pos] = 0
            U[pos + 1, pos] = U[pos, pos + 1] = 1
            U[pos + 1, pos + 1] = -q
    return U, (last_change >= 0)


def matrices_are_equal(A, B, epsilon=1e-6):
    """
    Return whether A and B are approximately equal, i.e.:

        ||A-B||_{infty} <= epsilon.
    """
    result = (abs(A - B) <= epsilon).all()
    if not result:
        print('Matrix A:', A, 'Matrix B:', B, sep='\n')
    return result


def is_qr_of(B, R):
    Rp = qr_decompose(B)
    are_eq = matrices_are_equal(Rp, R, 0.1)
    if not are_eq:
        print("Difference matrix:", Rp - R, "Gram matrix R:", R.transpose() @ R, "Gram matrix R':", Rp.transpose() @ Rp, sep="\n")
    return are_eq


def seysen_lll(B, args):
    """
    :param B: a basis, consisting of *column vectors*.
    :param delta: delta factor
    :return: transformation matrix U such that BU is LLL reduced.
    """
    global _delta, _c_dll

    delta, cores, lll_size = args.delta, args.cores, args.LLL

    n, is_modified, prof = len(B), True, TimeProfile()
    U = np.identity(n, dtype=np.int64)
    U_seysen = np.identity(n, dtype=np.float64)

    # Create the jobs:

    # 1) The C function supports block sizes up to 64.
    # 2) When a block size > n/2 is used, only 1 block is used, which is slow.
    block_size = min(lll_size, 64, n // 2)

    with Pool(cores) as p:
        while is_modified:
            t1 = perf_counter_ns()

            # Step 1: QR-decompose B_red, and only store the upper-triangular matrix R.
            R = qr_decompose(B @ U)

            t2 = perf_counter_ns()

            # Step 2: Call LLL concurrently on small blocks.
            if block_size > 2:
                offset = block_size//2 if prof.num_iterations % 2 == 1 else 0
                result = perform_lll_on_blocks(R, U, delta, offset, block_size)
                if not result:
                    print("An error occured in the C++ level. Aborting...")
                    exit(1)
                # LLL "destroys" the QR decomposition, so do it again.
                R = qr_decompose(B @ U)

            t3 = perf_counter_ns()

            # Step 3: Seysen reduce the upper-triangular matrix R.
            seysen_reduce(R, U_seysen)

            t4 = perf_counter_ns()

            # Step 4: If possible, perform Lagrange reduction on disjoint pairwise basis vectors.
            # Note: this step is negligible compared to the rest.
            U_lagrange, is_modified = lagrange_reduce(R @ U_seysen, delta)

            # Step 5: Update matrices with the transformation matrices from Step 3 & 4.
            with np.errstate(all='raise'):
                U_update = (U_seysen @ U_lagrange).astype(np.int64)
                U = U @ U_update

            t5 = perf_counter_ns()

            prof += TimeProfile.iteration(t1, t2, t3, t4, t5)
            print('.', end='', file=stderr)
            stderr.flush()
    return U, B @ U, prof
