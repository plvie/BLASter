"""
LLL reduction with Seysen instead of size reduction.
"""
from functools import partial
from sys import stderr
from time import perf_counter_ns

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, PillowWriter

from seysen_lll import set_debug_flag, set_num_cores, \
        block_lll, block_deep_lll, block_bkz, \
        ZZ_matmul_strided, ZZ_right_matmul, FT_matmul
from stats import get_profile, rhf, slope, potential


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
            # Reduce [i + hwidth, i + width) with respect to [i, i + hwidth).
            #
            #     [R11 R12]      [U11 U12]              [S11 S12]
            # R = [ 0  R22], U = [ 0  U22], S = R · U = [ 0  S22]
            #
            # The previous iteration has computed U11 and U22, so
            # Currently, R11 and R22 contain the values of
            # S11 = R11 · U11 and S22 = R22 · U22 respectively.
            j, k = i + hwidth, min(n, i + width)

            # S12' = R12 · U22.
            R[i:j, j:k] = FT_matmul(R[i:j, j:k], U[j:k, j:k].astype(np.float64))

            # W = round(S11^{-1} · S12').
            W = np.rint(FT_matmul(np.linalg.inv(R[i:j, i:j]), R[i:j, j:k]))

            # U12 = U11 · W
            U[i:j, j:k] = ZZ_matmul_strided(-U[i:j, i:j], W.astype(np.int64))

            # S12 = S12' - S11 · W.
            R[i:j, j:k] -= FT_matmul(R[i:j, i:j], W.astype(np.float64))

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
#        S11 = FT_matmul(R[:m, :m], U[:m, :m].astype(np.float64))
#        S12 = FT_matmul(R[:m, m:], U[m:, m:].astype(np.float64))
#
#        # W = round(S11^{-1} S12).
#        W = np.rint(FT_matmul(np.linalg.inv(S11), S12)).astype(np.int64)
#        # Now take the fractional part of the entries of W.
#        U[:m, m:] = ZZ_matmul_strided(-U[:m, :m], W)


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
    beta, num_tours = args.beta, args.num_tours  # BKZ params
    tours_done, cur_front = 0, 0

    set_num_cores(cores)
    set_debug_flag(args.profile)

    # Set up logfile
    logfile = args.logfile
    if logfile:
        logfile = open(logfile, "w")
        # TT: total wall time used by SeysenLLL
        print('it,   TT,       rhf,      slope,     potential', file=logfile)

    # Set up animation
    has_animation = bool(args.anim)
    if has_animation:
        fig, ax = plt.subplots()
        ax.set(xlim=[0, n])
        artists = []

    # Reduction function to call in each iteration:
    red_fn, red_char = block_lll, '.'  # LLL reduction
    if depth:
        red_fn = partial(block_deep_lll, depth)  # Deep-LLL
    elif beta:
        # In the literature on BKZ, it is usual to run LLL before calling the SVP oracle in BKZ.
        # However, it is actually better to preprocess the basis with DeepLLL-4 instead of LLL,
        # before calling the SVP oracle.
        red_fn = partial(block_deep_lll, 4)

    B_red = B.copy()
    U = np.identity(n, dtype=np.int64)
    U_seysen = np.identity(n, dtype=np.int64)

    tstart = perf_counter_ns()

    # Keep running until the basis is LLL reduced.
    # For BKZ: keep running until enumeration is called `num_tours` times.
    while tours_done < num_tours if beta else not is_reduced:
        # Step 1: QR-decompose B_red, and only store the upper-triangular matrix R.
        t1 = perf_counter_ns()
        R = np.linalg.qr(B_red, mode='r')
        if has_animation:
            artists.append(ax.plot(range(n), get_profile(R, True), color="blue"))

        # Step 2: Call LLL concurrently on small blocks.
        t2 = perf_counter_ns()
        offset = lll_size // 2 if tprof.num_iterations % 2 == 0 else 0

        if beta:
            red_char = 'E' if is_reduced else '.'
            if is_reduced:
                offset = (cur_front % lll_size)
                block_bkz(beta, R, B_red, U, delta, offset, lll_size)
                cur_front += (lll_size + 1 - (cur_front % 2)) // 2
                if cur_front >= n:
                    cur_front -= n
                    tours_done += 1
            else:
                # Perform `red_fn` before calling the enumeration code.
                red_fn(R, B_red, U, delta, offset, lll_size)
        else:
            red_fn(R, B_red, U, delta, offset, lll_size)  # LLL or Deep-LLL

        if args.profile:
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
        ZZ_right_matmul(U, U_seysen)
        ZZ_right_matmul(B_red, U_seysen)

        t6 = perf_counter_ns()

        tprof.tick(t2 - t1 + t4 - t3, t3 - t2, t5 - t4, t6 - t5)
        if verbose:
            print(red_char, end='', file=stderr, flush=True)
        if logfile is not None:
            TT = (t6 - tstart) * 10**-9
            prof = get_profile(R, True)
            print(f'{tprof.num_iterations:4d}, {TT:.6f}, {rhf(prof):8.6f}, {slope(prof):9.6f}, '
                  f'{potential(prof):9.3f}',
                  file=logfile)

        # Step 6: Check whether the basis is weakly-LLL reduced.
        is_reduced = is_weakly_lll_reduced(R, delta)

    # Close logfile
    if logfile:
        logfile.close()

    # Save and/or show the animation
    if has_animation:
        if verbose:
            print('\nOutputting animation...', file=stderr)
        fig.tight_layout()
        ani = ArtistAnimation(fig=fig, artists=artists, interval=200)
        # Generate 1920x1080 image:
        plt.gcf().set_size_inches(16, 9)
        # plt.show()
        ani.save(args.anim, dpi=120, writer=PillowWriter(fps=5))

    return U, B_red, tprof
