from math import log, prod, sqrt
from time import perf_counter_ns
import numpy as np


class TimeProfile:
    def __init__(self):
        self.num_iterations = 0
        self.time_qr = self.time_seysen = self.time_lagrange = self.time_matmul = 0

    def Iteration(t1, t2, t3, t4, t5):
        prof = TimeProfile() 
        prof.num_iterations = 1
        prof.time_qr = t2 - t1
        prof.time_seysen = t3 - t2
        prof.time_lagrange = t4 - t3
        prof.time_matmul = t5 - t4
        return prof
        

    def __iadd__(self, other):
        self.num_iterations += other.num_iterations
        self.time_qr += other.time_qr
        self.time_seysen += other.time_seysen
        self.time_lagrange += other.time_lagrange
        self.time_matmul += other.time_matmul
        return self

    def __str__(self):
        return (f"Iterations: {self.num_iterations}\n"
                f"Time QR factorization: {self.time_qr:18,d} ns\n"
                f"Time Seysen reduction: {self.time_seysen:18,d} ns\n"
                f"Time Lagrange reduct.: {self.time_lagrange:18,d} ns\n"
                f"Time Matrix Multipli.: {self.time_matmul:18,d} ns")


# Turns out, this is slower than `R = np.linalg.qr(B, 'r')`.
def block_cholesky(G, R):
    """
    Return the Cholesky decomposition of G using a recursive strategy.
    :param G: gram matrix of some basis.
    :param R: result R will be upper triangular satisfying: `R^T R = G`.
    :return: None! Result is stored in R.
    """
    if len(G) == 1:
        R[0, 0] = np.array([[sqrt(G[0, 0])]])
    else:
        n, m = len(G), len(G) // 2

        ################################################################################
        # Alternative, using fewer matrix multiplications:
        ################################################################################
        # Given B = [B0, B1], G = [[B0^T B0, B0^T B1], [B1^T B0, B1^T B1]].
        # Goal: find R = [[R0, S], [0, R1]] such that R^T R = G.
        # Line 3: Recover R0.
        block_cholesky(G[:m, :m], R[:m, :m])
        # Line 4: S satisfies: G01 = R0^T S --> (S = R0^{-T} G01)
        R[:m, m:] = np.linalg.inv(R[:m, :m]).transpose() @ G[:m, m:]
        # Line 5: Recover R1 = BlockCholesky(G11 - S^T S)
        block_cholesky(G[m:, m:] - R[:m, m:].transpose() @ R[:m, m:], R[m:, m:])
        return

        ################################################################################
        # Algorithm 6 from [KEF21] Towards Faster Polynomial-Time Lattice Reduction:
        ################################################################################
        ## Line 3: R_A <- BlockCholesky(A)
        # block_cholesky(G[:m, :m], R[:m, :m])
        ## Line 4:  R_A' <- Invert(R_A)
        # R0inv = np.linalg.inv(R[:m, :m])
        ## Line 5: A' <- R_A'^T R_A'
        ## However, there is an error here. It should be:
        ##         A' <- R_A' R_A'^T
        # Aprime = R0inv @ R0inv.transpose()
        ## Line 6: R_S <- BlockCholesky(C - B^T A' B)
        # block_cholesky(G[m:, m:] - G[m:, :m] @ Aprime @ G[:m, m:], R[m:, m:])
        ## Line 7: Return [[R_A, R_A'^T B], [0, R_S]]
        # R[:m, m:] = R0inv.transpose() @ G[:m, m:]


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
            # The disjoint lagrange reductions yield independent transformation matrices.
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


def seysen_lll(B, delta):
    """
    :param B: a basis, consisting of *column vectors*.
    :return: transformation matrix U such that BU is LLL reduced.
    """
    n = len(B)
    U, U1, Bred = np.identity(n, dtype=np.float64), np.zeros((n, n), dtype=np.float64), np.copy(B)
    prof = TimeProfile()
    is_modified = True
    while is_modified:
        t1 = perf_counter_ns()
        # Step 1: QR-decompose Bred, and only store the upper-triangular matrix R.
        R = np.linalg.qr(Bred, mode='r')  # block_cholesky(Bp.transpose() @ Bp, R)
        t2 = perf_counter_ns()
        # Step 2: Seysen reduce the upper-triangular matrix R.
        seysen_reduce(R, U1)
        t3 = perf_counter_ns()
        # Step 3: If possible, perform Lagrange reduction on disjoint pairwise basis vectors.
        U2, is_modified = lagrange_reduce(R @ U1, delta)
        t4 = perf_counter_ns()
        # Step 4: Update matrices with the transformation matrices from Step 3 & 4.
        U12 = U1 @ U2
        U = U @ U12
        Bred = Bred @ U12
        t5 = perf_counter_ns()

        prof += TimeProfile.Iteration(t1, t2, t3, t4, t5)
    return U, prof
