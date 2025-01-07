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
        block_lll, block_deep_lll, block_svp, \
        ZZ_matmul_strided, ZZ_right_matmul, FT_matmul
from stats import get_profile, rhf, slope, potential


class TimeProfile:
    """
    Object containing time spent on different parts within Seysen-LLL reduction.
    """

    def __init__(self):
        self._strs = ["QR-decomp.", "LLL-red.", "BKZ-red.", "Seysen-red.", "Matrix-mul."]
        self.num_iterations = 0
        self.times = [0] * 5

    def tick(self, *times):
        self.num_iterations += 1
        self.times = [x + y for x, y in zip(self.times, times)]

    def __str__(self):
        return (
            f"Iterations: {self.num_iterations}\n" +
           "\n".join(f"t_{{{s:11}}}={t/10**9:10.3f}s" for s, t in zip(self._strs, self.times) if t)
       )


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


def lll_reduce(B, U, U_seysen, lll_size, delta, depth,
               tprof, tracers, debug):
    """
    Perform Seysen + LLL-reduction on basis B, and keep track of the transformation in U.
    If `depth` is supplied, use deep insertions up to depth `depth`.
    """
    n, is_reduced, offset = B.shape[1], False, 0
    red_fn = partial(block_deep_lll, depth) if depth else block_lll

    # Keep running until the basis is LLL reduced.
    while not is_reduced:
        # Step 1: QR-decompose B, and only store the upper-triangular matrix R.
        t1 = perf_counter_ns()
        R = np.linalg.qr(B, mode='r')

        # Step 2: Call LLL concurrently on small blocks.
        t2 = perf_counter_ns()
        offset = lll_size // 2 if offset == 0 else 0
        red_fn(R, B, U, delta, offset, lll_size)  # LLL or Deep-LLL

        if debug:
            for i in range(offset, n, lll_size):
                j = min(n, i + lll_size)
                # Check whether R_[i:j) is really LLL-reduced.
                assert is_lll_reduced(R[i:j, i:j], delta)

        # Step 3: QR-decompose again because LLL "destroys" the QR decomposition.
        # Note: it does not destroy the bxb blocks, but everything above these: yes!
        t3 = perf_counter_ns()
        R = np.linalg.qr(B, mode='r')

        # Step 4: Seysen reduce the upper-triangular matrix R.
        t4 = perf_counter_ns()
        with np.errstate(all='raise'):
            seysen_reduce_iterative(R, U_seysen)

        # Step 5: Update B and U with transformation from Seysen.
        t5 = perf_counter_ns()
        ZZ_right_matmul(U, U_seysen)
        ZZ_right_matmul(B, U_seysen)

        # Step 6: Check whether the basis is weakly-LLL reduced.
        t6 = perf_counter_ns()

        is_reduced = is_weakly_lll_reduced(R, delta)
        tprof.tick(t2 - t1 + t4 - t3, t3 - t2, 0, t5 - t4, t6 - t5)

        # After time measurement:
        prof = get_profile(R, True)  # Seysen did not modify the diagonal of R
        note = (f"DeepLLL-{depth}" if depth else "LLL", None)
        for tracer in tracers.values():
            tracer(tprof.num_iterations, prof, note)


def bkz_reduce(B, U, U_seysen, lll_size, delta, depth,
               beta, bkz_tours, tprof, tracers, debug):
    """
    Perform Seysen + LLL-reduction on basis B, and keep track of the transformation in U.
    If `depth` is supplied, use deep insertions up to depth `depth`.
    """
    # BKZ parameters:
    n, tours_done, cur_front = B.shape[1], 0, 0

    lll_reduce(B, U, U_seysen, lll_size, delta, depth, tprof, tracers, debug)

    while tours_done < bkz_tours:
        # Step 1: QR-decompose B, and only store the upper-triangular matrix R.
        t1 = perf_counter_ns()
        R = np.linalg.qr(B, mode='r')

        # Step 2: Call BKZ concurrently on small blocks!
        t2 = perf_counter_ns()
        offset = (cur_front % beta)
        block_svp(beta, R, B, U, delta, offset)

        # Step 3: QR-decompose again because BKZ "destroys" the QR decomposition.
        # Note: it does not destroy the bxb blocks, but everything above these: yes!
        t3 = perf_counter_ns()
        R = np.linalg.qr(B, mode='r')

        # Step 4: Seysen reduce the upper-triangular matrix R.
        t4 = perf_counter_ns()
        with np.errstate(all='raise'):
            seysen_reduce_iterative(R, U_seysen)

        # Step 5: Update B and U with transformation from Seysen.
        t5 = perf_counter_ns()
        ZZ_right_matmul(U, U_seysen)
        ZZ_right_matmul(B, U_seysen)

        t6 = perf_counter_ns()

        tprof.tick(t2 - t1 + t4 - t3, 0, t3 - t2, t5 - t4, t6 - t5)

        # After time measurement:
        prof = get_profile(R, True)  # Seysen did not modify the diagonal of R
        note = (f"BKZ-{beta}", (beta, tours_done, bkz_tours, cur_front))
        for tracer in tracers.values():
            tracer(tprof.num_iterations, prof, note)

        # After printing: update the current location of the 'reduction front'
        cur_front += 1
        if cur_front + 2 >= n:
            # We are at the end of a tour.
            cur_front = 0
            tours_done += 1

        # Perform a final LLL reduction at the end
        lll_reduce(B, U, U_seysen, lll_size, delta, depth, tprof, tracers, debug, size_reduce)


def seysen_lll(
        B, lll_size: int = 64, delta: float = 0.99, cores: int = 1, debug: bool = False,
        verbose: bool = False, logfile: str = None, anim: str = None, depth: int = 0,
        size_reduce: bool = False,
        **kwds
):
    """
    :param B: a basis, consisting of *column vectors*.
    :param delta: delta factor for Lagrange reduction,
    :param cores: number of cores to use, and
    :param lll_size: the block-size for LLL, and
    :param debug: whether or not to debug and print more output on time consumption.
    :param kwds: additional arguments (for BKZ reduction).

    :return: tuple (U, B · U, tprof) where:
        U: the transformation matrix such that B · U is LLL reduced,
        B · U: an LLL-reduced basis,
        tprof: TimeProfile object.
    """
    n, tprof = B.shape[1], TimeProfile()
    lll_size = min(max(2, lll_size), n)

    set_num_cores(cores)
    set_debug_flag(1 if debug else 0)

    tracers = {}
    if verbose:
        def trace_print(_, prof, note):
            if note[0].startswith('BKZ'):
                beta, tour, ntours, touridx = note[1]
                if touridx % 10 == 0:
                    print(f"\nBKZ(β:{beta:3d},t:{tour + 1:2d}/{ntours:2d}, o:{touridx:4d}): "
                          f"slope={slope(prof):.6f}, rhf={rhf(prof):.6f}",
                          end="", file=stderr, flush=True)
            else:
                print('.', end="", file=stderr, flush=True)

        tracers['v'] = trace_print

    # Set up logfile
    has_logfile = logfile is not None
    if has_logfile:
        tstart = perf_counter_ns()
        logfile = open(logfile, "w", encoding="utf8")
        print('it,walltime,rhf,slope,potential,note', file=logfile, flush=True)

        def trace_logfile(it, prof, note):
            walltime = (perf_counter_ns() - tstart) * 10**-9
            print(f'{it:4d},{walltime:.6f},{rhf(prof):8.6f},{slope(prof):9.6f},'
                  f'{potential(prof):9.3f},{note[0]}', file=logfile)

        tracers['l'] = trace_logfile

    # Set up animation
    has_animation = anim is not None
    if has_animation:
        fig, ax = plt.subplots()
        ax.set(xlim=[0, n])
        artists = []

        def trace_anim(it, prof, _):
            artists.append(ax.plot(range(n), prof, color="blue"))

        tracers['a'] = trace_anim

    B = B.copy()  # Do not modify B in-place, but work with a copy.
    U = np.identity(n, dtype=np.int64)
    U_seysen = np.identity(n, dtype=np.int64)

    beta = kwds.get("beta")
    try:
        if not beta:
            lll_reduce(B, U, U_seysen, lll_size, delta, depth, tprof, tracers, debug)
        else:
            # Parse BKZ parameters:
            bkz_tours = kwds.get("bkz_tours") or 1
            bkz_prog = kwds.get("bkz_prog") or beta

            # Progressive-BKZ: start running BKZ-beta' for some `beta' >= 40`,
            # then increase the blocksize beta' by `bkz_prog` and run BKZ-beta' again,
            # and repeat this until `beta' = beta`.
            betas = range(40 + ((beta - 40) % bkz_prog), beta + 1, bkz_prog)

            # In the literature on BKZ, it is usual to run LLL before calling the SVP oracle in BKZ.
            # However, it is actually better to preprocess the basis with DeepLLL-4 instead of LLL,
            # before calling the SVP oracle.
            for beta_ in betas:
                bkz_reduce(B, U, U_seysen, lll_size, delta, 4, beta_,
                           bkz_tours if beta_ == beta else 1, tprof, tracers, debug)
    except KeyboardInterrupt:
        pass  # When interrupted, give the partially reduced basis.

    # Close logfile
    if has_logfile:
        logfile.close()

    # Save and/or show the animation
    if has_animation:
        # Saving the animation takes a LONG time.
        if verbose:
            print('\nOutputting animation...', file=stderr)
        fig.tight_layout()
        ani = ArtistAnimation(fig=fig, artists=artists, interval=200)
        # Generate 1920x1080 image:
        plt.gcf().set_size_inches(16, 9)
        # plt.show()
        ani.save(anim, dpi=120, writer=PillowWriter(fps=5))

    return U, B, tprof
