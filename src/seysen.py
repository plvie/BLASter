"""
LLL reduction with Seysen instead of size reduction.
"""

from math import sqrt, log
from sys import stderr
from time import perf_counter_ns
import numpy as np
from threadpoolctl import threadpool_limits

from seysen_lll import perform_lll_on_blocks, eigen_init, eigen_matmul, eigen_right_matmul


def potential(B):
    profile = [log(abs(d_i)) for d_i in np.linalg.qr(B, mode='r').diagonal()]
    n = len(profile)
    return sum((n - i) * profile[i] for i in range(n))


has_eigenpy = True
try:
    from eigenpy import LLT
except ImportError:
    # print('Library `eigenpy` is not found.')
    has_eigenpy = False

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
        if has_eigenpy:
            # return eigen_cholesky(Bf.transpose() @ Bf)
            return eigen_cholesky(B.transpose() @ B)
        else:
            Bf = B.astype(np.float64)
            R = np.identity(B.shape[1], dtype=np.float64)
            block_cholesky(Bf.transpose() @ Bf, R)
            return R
    else:
        R = np.linalg.qr(B, mode='r')
        for i in range(B.shape[1]):
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

        S11 = R[:m, :m] @ U[:m, :m].astype(np.float64)
        S12 = R[:m, m:] @ U[m:, m:].astype(np.float64)

        # W = round(S11^{-1} S12).
        W = np.rint(np.linalg.inv(S11) @ S12).astype(np.int64)
        # Now take the fractional part of the entries of W.
        U[:m, m:] = eigen_matmul(np.ascontiguousarray(-U[:m, :m]), W)


def is_weakly_lll_reduced(R, delta=.99):
    """
    Return whether R is Weakly-LLL reduced
    :param R: upper-triangular matrix
    :param delta: delta-factor used in the Lovasz condition
    :return: bool
    """
    n = len(R)
    for pos in range(0, n - 1):
        # vector b0 = (u, 0)
        u = abs(R[pos, pos])
        # vector b1 = (v, w)
        v, w = R[pos, pos + 1], R[pos + 1, pos + 1]
        v_mod = ((v + u/2) % u) - u/2

        if v_mod**2 + w**2 <= delta * u**2:
            return False
    return True


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
    :param args: arguments containing:
        - delta: delta factor for Lagrange reduction,
        - cores: number of cores to use, and
        - LLL:   the block-size for LLL.
    :return: tuple (U, B @ U, profile) where:
        U: the transformation matrix such that B @ U is LLL reduced,
        B @ U: an LLL-reduced basis,
        profile: TimeProfile object.
    """
    n, is_reduced, prof = B.shape[1], False, TimeProfile()
    delta, cores, lll_size = args.delta, args.cores, min(max(2, args.LLL), n)
    B_red = B.copy()
    U = np.identity(n, dtype=np.int64)
    U_seysen = np.identity(n, dtype=np.int64)

    eigen_init(cores)

    while not is_reduced:
        t1 = perf_counter_ns()

        # Step 1: QR-decompose B_red, and only store the upper-triangular matrix R.
        R = qr_decompose(B_red)

        t2 = perf_counter_ns()

        # Step 2: Call LLL concurrently on small blocks.
        offset = lll_size//2 if prof.num_iterations % 2 == 1 else 0
        result = perform_lll_on_blocks(R, U, delta, offset, lll_size)
        if not result:
            print("An error occured in the C++ level. Aborting...")
            exit(1)

        B_red = eigen_matmul(B, U)  # B @ U
        # LLL "destroys" the QR decomposition, so do it again.
        R = qr_decompose(B_red)

        t3 = perf_counter_ns()

        # Step 3: Seysen reduce the upper-triangular matrix R.
        with np.errstate(all='raise'), threadpool_limits(limits=1):
            seysen_reduce(R, U_seysen)

        t4 = perf_counter_ns()

        # Step 4: If possible, perform Lagrange reduction on disjoint pairwise basis vectors.
        # Note: this step is negligible compared to the rest.
        # Note: we multiply R and U_seysen using BLAS in numpy.
        is_reduced = is_weakly_lll_reduced(R @ U_seysen, delta)

        # Step 5: Update matrices with the transformation matrices from Step 3 & 4.
        with np.errstate(all='raise'):
            eigen_right_matmul(U, U_seysen)
            eigen_right_matmul(B_red, U_seysen)

        t5 = perf_counter_ns()

        prof += TimeProfile.iteration(t1, t2, t3, t4, t5)
        print('.', end='', file=stderr)
        stderr.flush()
    return U, B_red, prof
