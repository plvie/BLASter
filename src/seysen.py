"""
LLL reduction with Seysen instead of size reduction.
"""

from sys import stderr
from time import perf_counter_ns

import numpy as np

from seysen_lll import (
    block_lll, block_deep_lll, block_bkz,
    eigen_init, eigen_matmul, eigen_right_matmul,
)
from stats import get_profile, rhf, slope, potential

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class TimeProfile:
    """
    Object containing time spent on different parts within Seysen-LLL reduction.
    """

    def __init__(self):
        self.num_iterations = 0
        self.time_qr = self.time_lll = self.time_seysen = self.time_matmul = 0

    def tick(self, t_qr, t_lll, t_seysen, t_matmul):
        self.num_iterations += 1
        self.time_qr += t_qr
        self.time_lll += t_lll
        self.time_seysen += t_seysen
        self.time_matmul += t_matmul

    def __str__(self):
        return (f"Iterations: {self.num_iterations}\n"
                f"Time QR factorization: {self.time_qr:18,d} ns\n"
                f"Time LLL    reduction: {self.time_lll:18,d} ns\n"
                f"Time Seysen reduction: {self.time_seysen:18,d} ns\n"
                f"Time Matrix Multipli.: {self.time_matmul:18,d} ns")


def float_matmul(A, B):
    # Note: NumPy uses BLAS to multiply floating-point matrices.
    return A @ B


def seysen_reduce_iterative(R, U):
    """
    Perform Seysen reduction on a matrix R, while keeping track of the transformation matrix U.
    The matrix R is updated along the way.

    :param R: an upper-triangular matrix that will be modified
    :param U: an upper-triangular transformation matrix such that diag(U) = (1, 1, ..., 1).
    :return: Nothing! After termination, R is Seysen reduced.
    """
    # Assume diag(U) = (1, 1, ..., 1).
    n = len(R)
    for i in range(0, n-1, 2):
        U[i, i + 1] = -round(R[i, i + 1] / R[i, i])
        R[i, i + 1] += R[i, i] * U[i, i + 1]

    width, hwidth = 4, 2
    while hwidth < n:
        for i in range(0, n - hwidth, width):
            # Reduce [i + hwidth, i + width) with respect to [i, i + hwidth)
            j, k = i + hwidth, min(n, i + width)

            # S11 = R11 · U11, S12' = R12 · U22, S22 = R22 · U22.
            R[i:j, j:k] = float_matmul(R[i:j, j:k], U[j:k, j:k].astype(np.float64))
            # W = round(S11^{-1} S12').
            W = np.rint(float_matmul(np.linalg.inv(R[i:j, i:j]), R[i:j, j:k]))

            # U12 = U11 · W
            U[i:j, j:k] = eigen_matmul(np.ascontiguousarray(-U[i:j, i:j]), W.astype(np.int64))
            # S12 = S12' - S11 · W.
            R[i:j, j:k] -= float_matmul(R[i:j, i:j], W.astype(np.float64))
        width, hwidth = 2 * width, width


# def seysen_reduce(R, U):
#    """
#    Seysen reduce a matrix R, recursive style, and store the result in U.
#    See: Algorithm 7 from [KEF21].
#    [KEF21] P. Kircher, T. Espitau, P.-A. Fouque. Towards faster
#    polynomial-time lattice reduction.
#    :param R: an upper-triangular matrix (having row vectors).
#    :param U: a unimodular transformation U such that RU is Seysen-Reduced.
#    :return: None! The result is stored in U.
#    """
#    n, m = len(R), len(R) // 2
#    # TODO: Write an iterative version that beats the recursive version.
#
#    if n == 1:
#        # Base case
#        U[0, 0] = 1
#    elif n == 2:
#        # Make sure RU is size-reduced, i.e. |R00*X + R01| <= |R00|/2
#        U[0, 0] = U[1, 1] = 1
#        U[0, 1] = -round(R[0, 1] / R[0, 0])
#    else:
#        # R11, R12, R22 = R[:m, :m], R[:m, m:], R[m:, m:]
#        seysen_reduce(R[:m, :m], U[:m, :m])
#        seysen_reduce(R[m:, m:], U[m:, m:])
#
#        S11 = float_matmul(R[:m, :m], U[:m, :m].astype(np.float64))
#        S12 = float_matmul(R[:m, m:], U[m:, m:].astype(np.float64))
#
#        # W = round(S11^{-1} S12).
#        W = np.rint(float_matmul(np.linalg.inv(S11), S12)).astype(np.int64)
#        # Now take the fractional part of the entries of W.
#        U[:m, m:] = eigen_matmul(np.ascontiguousarray(-U[:m, :m]), W)


def is_weakly_lll_reduced(R, delta=.99):
    """
    Return whether R is Weakly-LLL-reduced
    :param R: upper-triangular matrix
    :param delta: delta-factor used in the Lovasz condition
    :return: bool
    """
    n = len(R)
    for pos in range(0, n - 1):
        # vectors are b0 = (u, 0), b1 = (v, w).
        u = abs(R[pos, pos])
        v, w = R[pos, pos + 1], R[pos + 1, pos + 1]
        v_mod = ((v + u/2) % u) - u/2

        if v_mod**2 + w**2 <= delta * u**2:
            return False  # ||b1||^2 <= delta ||b0||^2
    return True


def is_size_reduced(R):
    """
    Return whether R is size-reduced.
    :param R: upper-triangular matrix
    :return: bool
    """
    return all(max(abs(R[i, i + 1:])) <= abs(R[i, i]) / 2 for i in range(len(R) - 1))


def is_lll_reduced(R, delta=.99):
    """
    Return whether R is LLL-reduced
    :param R: upper-triangular matrix
    :param delta: delta-factor used in the Lovasz condition
    :return: bool
    """
    return is_weakly_lll_reduced(R, delta) and is_size_reduced(R)


def seysen_lll(B, args):
    """
    :param B: a basis, consisting of *column vectors*.
    :param args: arguments containing:
        - delta: delta factor for Lagrange reduction,
        - cores: number of cores to use, and
        - LLL:   the block-size for LLL.
    :return: tuple (U, B · U, profile) where:
        U: the transformation matrix such that B · U is LLL reduced,
        B · U: an LLL-reduced basis,
        profile: TimeProfile object.
    """
    n, is_reduced, tprof = B.shape[1], False, TimeProfile()

    # Parse all arguments.
    # TODO: work with **kwargs?
    delta, cores, verbose = args.delta, args.cores, args.verbose
    lll_size = min(max(2, args.LLL), n)
    depth = args.depth  # Deep-LLL params
    beta, max_enum_calls, enum_calls = args.beta, args.max_tours, 0  # BKZ params
    if not max_enum_calls:
        # This corresponds to *8* `original` BKZ-tours,
        # because one enumeration call calls SVP on `lll_size/2` consecutive positions.
        max_enum_calls = 16 * n / lll_size

    # set up logfile
    logfile = args.logfile
    if logfile:
        logfile = open(logfile, "w")
        # TT: total wall time used by SeysenLLL
        print('it,   TT,       rhf,      slope,     potential', file=logfile)

    # Set up animation
    fig, ax = plt.subplots()
    ax.set(xlim=[0, n])
    artists = []

    B_red = B.copy()
    U = np.identity(n, dtype=np.int64)
    U_seysen = np.identity(n, dtype=np.int64)

    eigen_init(cores)

    tstart = perf_counter_ns()
    # Keep running until the basis is LLL reduced.
    # For BKZ: keep running until enumeration is called `max_enum_calls` times.
    while (enum_calls < max_enum_calls) if beta else (not is_reduced):
        # Step 1: QR-decompose B_red, and only store the upper-triangular matrix R.
        t1 = perf_counter_ns()
        R = np.linalg.qr(B_red, mode='r')
        artists.append(ax.plot(range(n), [log2(abs(x)) for x in R.diagonal()], color="blue"))

        # Step 2: Call LLL concurrently on small blocks.
        t2 = perf_counter_ns()
        offset = 0 if tprof.num_iterations % 2 == 0 else lll_size//2

        if depth:
            block_deep_lll(R, B_red, U, delta, offset, lll_size, depth)  # Deep-LLL
        elif beta:
            if is_reduced:
                offset = 0 if enum_calls % 2 == 0 else lll_size//2
                print('E', end='', file=stderr, flush=True)
                block_bkz(R, B_red, U, delta, offset, lll_size, beta, 1)  # BKZ
                enum_calls += 1
            else:
                # Perform global LLL reduction before calling the enumeration code.
                block_lll(R, B_red, U, delta, offset, lll_size)  # LLL
        else:
            block_lll(R, B_red, U, delta, offset, lll_size)  # LLL

        # TODO: remove this sanity-check to be quicker.
        for i in range(offset, n, lll_size):
            # Check whether R_[i:j) is really LLL-reduced.
            j = min(n, i + lll_size)
            assert is_lll_reduced(R[i:j, i:j], delta)

        # Step 3: QR-decompose again because LLL "destroys" the QR decomposition.
        # Note: it does not destroy the bxb blocks, but everything above these: yes!
        t3 = perf_counter_ns()
        R = np.linalg.qr(B_red, mode='r')

        # Step 4: Seysen reduce the upper-triangular matrix R.
        t4 = perf_counter_ns()
        with np.errstate(all='raise'):
            seysen_reduce_iterative(R, U_seysen)

        # Step 5: Update B_red and U with transformation from Seysen.
        t5 = perf_counter_ns()
        with np.errstate(all='raise'):
            eigen_right_matmul(U, U_seysen)
            eigen_right_matmul(B_red, U_seysen)

        t6 = perf_counter_ns()

        tprof.tick(t2 - t1 + t4 - t3, t3 - t2, t5 - t4, t6 - t5)
        if verbose:
            print('.', end='', file=stderr, flush=True)
        if logfile is not None:
            TT = (t6 - tstart) * 10**-9
            prof = get_profile(R)
            print(f'{tprof.num_iterations:4d}, {TT:.6f}, {rhf(prof):8.6f}, {slope(prof):9.6f}, '
                  f'{potential(prof):9.3f}',
                  file=logfile)

        # Step 6: Check whether the basis is weakly-LLL reduced.
        is_reduced = is_weakly_lll_reduced(R, delta)

    # Close logfile
    if logfile:
        logfile.close()

    # Show the animation
    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=400)
    plt.show()

    return U, B_red, tprof
