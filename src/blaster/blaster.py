"""
BLASter lattice reduction: LLL with QR decomposition, Seysen's reduction, and
segments, in which lattice reduction is done in parallel.
"""
from functools import partial
from sys import stderr
from time import perf_counter_ns

import numpy as np
import cupy as cp
from size_reduction_gpu import is_weakly_lll_reduced_gpu, seysen_reduce_gpu

# Local imports
from blaster_core import \
    set_debug_flag, set_num_cores, block_lll, block_deep_lll, block_bkz, ZZ_right_matmul ,get_R_sub_HKZ, get_G_sub_HKZ, apply_U_HKZ, block_lll_gpu, block_deep_lll_gpu, block_bkz_gpu
from size_reduction import is_lll_reduced, is_weakly_lll_reduced, size_reduce, seysen_reduce
from stats import get_profile, rhf, slope, potential, get_profile_gpu
from lattice_io import write_lattice

from hkz import hkz_kernel

class TimeProfile:
    """
    Object containing time spent on different parts when running BLASter.
    """

    def __init__(self, use_seysen: bool = False):
        self._strs = [
            "QR-decomp.", "LLL-red.", "BKZ-red.",
            "Seysen-red." if use_seysen else "Size-red.  ", "Matrix-mul."
        ]
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


def lll_reduce(B, U, U_seysen, lll_size, delta, depth,
               tprof, tracers, debug, use_seysen):
    """
    Perform BLASter's lattice reduction on basis B, and keep track of the transformation in U.
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
        red_fn(R, B, U, delta, offset, lll_size)  # LLL or deep-LLL

        if debug:
            for i in range(offset, n, lll_size):
                j = min(n, i + lll_size)
                # Check whether R_[i:j) is really LLL-reduced.
                assert is_lll_reduced(R[i:j, i:j], delta)

        # Step 3: QR-decompose again because LLL "destroys" the QR decomposition.
        # Note: it does not destroy the bxb blocks, but everything above these: yes!
        t3 = perf_counter_ns()
        R = np.linalg.qr(B, mode='r')

        # Step 4: Seysen reduce or size reduce the upper-triangular matrix R.
        t4 = perf_counter_ns()
        with np.errstate(all='raise'):
            (seysen_reduce if use_seysen else size_reduce)(R, U_seysen)
        # Step 5: Update B and U with transformation from Seysen's reduction.
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




def lll_reduce_gpu(B, U, U_seysen, lll_size, delta, depth,
               tprof, tracers, debug, use_seysen):
    """
    Perform BLASter's lattice reduction on basis B, and keep track of the transformation in U.
    If `depth` is supplied, use deep insertions up to depth `depth`.
    """
    n, is_reduced, offset = B.shape[1], 0, 0


    host_flag = cp.cuda.alloc_pinned_memory(1)

    use_gpu_lll = False

    is_B_gpu = isinstance(B, cp.ndarray)
    is_U_gpu = isinstance(U, cp.ndarray)
    is_U_s_gpu = isinstance(U_seysen, cp.ndarray)

    # 2) …and lift any host arrays up to the device once
    B_gpu        = B        if is_B_gpu   else cp.asarray(B)
    U_gpu        = U        if is_U_gpu   else cp.asarray(U)
    U_s_gpu      = U_seysen if is_U_s_gpu else cp.asarray(U_seysen)


    # 3) pick your GPU‐enabled block‐LLL routine # to do
    #red_fn = partial(block_deep_lll, depth) if depth else block_lll
    red_fn = partial(block_deep_lll_gpu, depth) if depth else block_lll_gpu
    pinned_host_arr = cp.cuda.alloc_pinned_memory(B_gpu.shape[0]* B_gpu.shape[1]* 8) #memory needed

    # On wrappe ça en numpy pour manipuler :
    R_cpu = np.frombuffer(pinned_host_arr, dtype=np.float64).reshape(B_gpu.shape)

    logging = False
    while not is_reduced:
        # — Step 1: QR on GPU
        t1 = perf_counter_ns()
        R_gpu = cp.linalg.qr(B_gpu, mode='r')
        # — Step 2: small‐block LLL on GPU (further work)
        t2 = perf_counter_ns()
        offset = lll_size//2 if offset==0 else 0
        if use_gpu_lll:  # once you have a GPU version
            # red_fn_gpu(R_gpu, B_gpu, U_gpu, delta, offset, lll_size)
            raise "Not implemented yet"
        else:
            # CPU fallback for testing
            # 1) transfer back to host
            cp.asnumpy(R_gpu, out=R_cpu) # synchro CPU-GPU so this is the time to compute also the other stuff

            # 2) run existing CPU version
            U_sub = red_fn(R_cpu, delta, offset, lll_size) # num_blocks, block_size**2

            #copy to the GPU
            U_sub_gpu = cp.asarray(U_sub)

            # Calcul du nombre de blocs
            num_blocks = int((n - offset + lll_size - 1) // lll_size)

            # Matrice GPU vide où on va copier les sous-blocs
            U_sub_total = cp.eye(n, dtype=U_sub_gpu.dtype)
            for block_id in range(num_blocks):
                # position de départ du bloc
                i = offset + lll_size * block_id
                # largeur effective du bloc (dernier bloc plus petit si besoin)
                w = min(n - i, lll_size)

                # on reshape les w*w premières valeurs en (w, w)
                block_vals = U_sub_gpu[block_id, : w*w].reshape(w, w)

                # on copie ce bloc dans U_sub_total
                U_sub_total[i : i + w, i : i + w] = block_vals

            # U_sub_total contient maintenant tous les sous-blocs copiés à leur place

            # 3) transfer results back to GPU (just U_sub and apply here U_sub to B and U)
            U_gpu = U_gpu @ U_sub_total
            B_gpu = B_gpu @ U_sub_total
            

        # — Step 3: re‐QR
        t3 = perf_counter_ns()
        R_gpu = cp.linalg.qr(B_gpu, mode='r')

        # — Step 4: Seysen or size‐reduce on GPU
        t4 = perf_counter_ns()
        if use_seysen:
            seysen_reduce_gpu(R_gpu, U_s_gpu)
        else:
            raise "Error not Implemented"

        # — Step 5: update your basis and U on GPU
        t5 = perf_counter_ns()
        U_gpu = U_gpu @ U_s_gpu
        B_gpu = B_gpu @ U_s_gpu

        # — Step 6: check “weak LLL” on GPU (or pull diag back)
        t6 = perf_counter_ns()


        is_weakly_lll_reduced_gpu(R_gpu, host_flag, delta)
        

        # profiling & tracing
        tprof.tick(t2-t1 + t4-t3, t3-t2, 0, t5-t4, t6-t5)
        if logging:
            prof = get_profile_gpu(R_gpu, True).get()
            note = (f"DeepLLL-{depth}" if depth else "LLL", None)
            for tracer in tracers.values():
                tracer(tprof.num_iterations, prof, note)

        
        mv = memoryview(host_flag)
        is_reduced = mv[0]

    # 4) finally, write results back to host arrays if needed
    if not is_U_gpu:
        U[:] = cp.asnumpy(U_gpu)
    if not is_B_gpu:
        B[:] = cp.asnumpy(B_gpu)
    if not is_U_s_gpu:
        U_seysen[:] = cp.asnumpy(U_s_gpu)
    
def bkz_reduce_gpu(B, U, U_seysen, lll_size, delta, depth,
               beta, bkz_tours, bkz_size, tprof, tracers, debug, use_seysen):
    
    # BKZ parameters:
    n, tours_done, cur_front = B.shape[1], 0, 0

    is_B_gpu = isinstance(B, cp.ndarray)
    is_U_gpu = isinstance(U, cp.ndarray)
    is_U_s_gpu = isinstance(U_seysen, cp.ndarray)

    # 2) …and lift any host arrays up to the device once
    B_gpu        = B        if is_B_gpu   else cp.asarray(B)
    U_gpu        = U        if is_U_gpu   else cp.asarray(U)
    U_s_gpu      = U_seysen if is_U_s_gpu else cp.asarray(U_seysen)

    lll_reduce_gpu(B_gpu, U_gpu, U_s_gpu, lll_size, delta, depth, tprof, tracers, debug, use_seysen)

    while tours_done < bkz_tours:
        # Step 1: QR-decompose B, and only store the upper-triangular matrix R.
        t1 = perf_counter_ns()
        # R = np.linalg.qr(B, mode='r')
        R_gpu = cp.linalg.qr(B_gpu, mode='r')

        # Step 2: Call BKZ concurrently on small blocks!
        t2 = perf_counter_ns()
        print("(BKZ) we are at :", cur_front)
        # norm_before = abs(R[cur_front, cur_front])

        R_cpu = cp.asnumpy(R_gpu)
        offset = cur_front % beta
        U_sub = block_bkz_gpu(beta, R_cpu, delta, offset, bkz_size)
        U_sub_gpu = cp.asarray(U_sub)
        bs         = bkz_size                # taille de bloc passée à block_bkz_gpu
        num_blocks = (n - offset + bs - 1) // bs

        # nombre de blocs pleins (taille exactly bs)
        num_full = (n - offset) // bs

        # 1) crée et initialise U_sub_total hors de toute boucle
        U_sub_total = cp.eye(n, dtype=U_sub_gpu.dtype)

        # 4) Boucle simple : reshape & copie bloc par bloc
        for block_id in range(num_blocks):
            # position du bloc sur la diagonale
            i = offset + block_id * bs
            # taille effective du bloc (dernier bloc possiblement plus petit)
            w = min(n - i, bs)
            # reshape des w*w premières valeurs de U_sub_gpu[block_id]
            block_vals = U_sub_gpu[block_id, : w*w].reshape(w, w)
            # copie sur la diagonale de U_sub_total
            U_sub_total[i : i + w, i : i + w] = block_vals

        # 4) application
        U_gpu = U_gpu @ U_sub_total
        B_gpu = B_gpu @ U_sub_total

        

        # Step 3: QR-decompose again because BKZ "destroys" the QR decomposition.
        # Note: it does not destroy the bxb blocks, but everything above these: yes!
        t3 = perf_counter_ns()
        R_gpu = cp.linalg.qr(B_gpu, mode='r')
        # print(abs(R[cur_front, cur_front]), norm_before)
        # assert abs(R[cur_front, cur_front]) <= norm_before
        # Step 4: Seysen reduce or size reduce the upper-triangular matrix R.
        t4 = perf_counter_ns()
        if use_seysen:
            seysen_reduce_gpu(R_gpu, U_s_gpu)
        else:
            raise "Error not Implemented"

        # Step 5: Update B and U with transformation from Seysen's reduction.
        t5 = perf_counter_ns()
        U_gpu = U_gpu @ U_s_gpu
        B_gpu = B_gpu @ U_s_gpu

        t6 = perf_counter_ns()

        tprof.tick(t2 - t1 + t4 - t3, 0, t3 - t2, t5 - t4, t6 - t5)

        # After time measurement:
        prof = get_profile_gpu(R_gpu, True).get()
        note = (f"DeepLLL-{depth}" if depth else "LLL", None)
        for tracer in tracers.values():
            tracer(tprof.num_iterations, prof, note)
        # After printing: update the current location of the 'reduction front'
        if cur_front + beta > n:
            # HKZ-reduction was performed at the end, which is the end of a tour.
            cur_front = 0
            tours_done += 1
        else:
            cur_front += (bkz_size - beta + 1)
            #cur_front += 1

        # Perform a final LLL reduction at the end
        lll_reduce_gpu(B_gpu, U_gpu, U_s_gpu, lll_size, delta, depth, tprof, tracers, debug, use_seysen)
               

def hkz_reduce(B, U, U_seysen, lll_size, delta, depth,
               beta, bkz_tours, block_size, tprof, tracers, debug, use_seysen, pump_and_jump):
    """
    Perform BLASter's BKZ reduction on basis B, and keep track of the transformation in U.
    If `depth` is supplied, BLASter's deep-LLL is called in between calls of the SVP oracle.
    Otherwise BLASter's LLL is run.
    """
    # BKZ parameters:
    n, tours_done, cur_front = B.shape[1], 0, 0

    lll_reduce_gpu(B, U, U_seysen, lll_size, delta, depth, tprof, tracers, debug, use_seysen)

    if block_size < beta:
        block_size = beta
    while tours_done < bkz_tours:
        #the best is to call hkz with the same blocksize as beta
        print("(HKZ) we are at :", cur_front)
        
        # Step 1: QR-decompose B, and only store the upper-triangular matrix R.
        t1 = perf_counter_ns()
        R = np.linalg.qr(B, mode='r')
        # Step 2: Call HKZ on small blocks!
        t2 = perf_counter_ns()

        w = block_size if (n - cur_front) >= block_size else (n - cur_front)
        R_sub = get_R_sub_HKZ(R, cur_front, w)
        U_sub = hkz_kernel(R_sub, w, beta, pump_and_jump)
        apply_U_HKZ(B, U, U_sub, cur_front, w)

        t3 = perf_counter_ns()
        R = np.linalg.qr(B, mode='r')
        # assert abs(R[cur_front, cur_front]) <= norm_before
        # Step 4: Seysen reduce or size reduce the upper-triangular matrix R.
        t4 = perf_counter_ns()
        with np.errstate(all='raise'):
            (seysen_reduce if use_seysen else size_reduce)(R, U_seysen)

        # Step 5: Update B and U with transformation from Seysen's reduction.
        t5 = perf_counter_ns()
        ZZ_right_matmul(U, U_seysen)
        ZZ_right_matmul(B, U_seysen)

        t6 = perf_counter_ns()

        tprof.tick(t2 - t1 + t4 - t3, 0, t3 - t2, t5 - t4, t6 - t5)

        # After time measurement:
        prof = get_profile(R, True)  # Seysen did not modify the diagonal of R
        note = (f"HKZ-{beta}", (block_size, tours_done, bkz_tours, cur_front))
        for tracer in tracers.values():
            tracer(tprof.num_iterations, prof, note)

        # After printing: update the current location of the 'reduction front'
        if cur_front + (block_size) >= n:
            # HKZ-reduction was performed at the end, which is the end of a tour.
            cur_front = 0
            tours_done += 1
        else:
            cur_front += (block_size - beta + 1)
        # Perform a final LLL reduction at the end
        lll_reduce_gpu(B, U, U_seysen, lll_size, delta, depth, tprof, tracers, debug, use_seysen)

def bkz_reduce(B, U, U_seysen, lll_size, delta, depth,
               beta, bkz_tours, bkz_size, tprof, tracers, debug, use_seysen):
    """
    Perform BLASter's BKZ reduction on basis B, and keep track of the transformation in U.
    If `depth` is supplied, BLASter's deep-LLL is called in between calls of the SVP oracle.
    Otherwise BLASter's LLL is run.
    """
    # BKZ parameters:
    n, tours_done, cur_front = B.shape[1], 0, 0

    lll_reduce_gpu(B, U, U_seysen, lll_size, delta, depth, tprof, tracers, debug, use_seysen)

    while tours_done < bkz_tours:
        # Step 1: QR-decompose B, and only store the upper-triangular matrix R.
        t1 = perf_counter_ns()
        R = np.linalg.qr(B, mode='r')

        # Step 2: Call BKZ concurrently on small blocks!
        t2 = perf_counter_ns()
        print("(BKZ) we are at :", cur_front)
        # norm_before = abs(R[cur_front, cur_front])
        block_bkz(beta, R, B, U, delta, cur_front % beta, bkz_size)
        

        # Step 3: QR-decompose again because BKZ "destroys" the QR decomposition.
        # Note: it does not destroy the bxb blocks, but everything above these: yes!
        t3 = perf_counter_ns()
        R = np.linalg.qr(B, mode='r')
        # print(abs(R[cur_front, cur_front]), norm_before)
        # assert abs(R[cur_front, cur_front]) <= norm_before
        # Step 4: Seysen reduce or size reduce the upper-triangular matrix R.
        t4 = perf_counter_ns()
        with np.errstate(all='raise'):
            (seysen_reduce if use_seysen else size_reduce)(R, U_seysen)

        # Step 5: Update B and U with transformation from Seysen's reduction.
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
        if cur_front + beta > n:
            # HKZ-reduction was performed at the end, which is the end of a tour.
            cur_front = 0
            tours_done += 1
        else:
            cur_front += (bkz_size - beta + 1)
            #cur_front += 1

        # Perform a final LLL reduction at the end
        lll_reduce_gpu(B, U, U_seysen, lll_size, delta, depth, tprof, tracers, debug, use_seysen)


def reduce(
        B, lll_size: int = 64, delta: float = 0.99, cores: int = 1, debug: bool = False,
        verbose: bool = False, logfile: str = None, anim: str = None, depth: int = 0,
        use_seysen: bool = False,
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
    n, tprof = B.shape[1], TimeProfile(use_seysen)
    lll_size = min(max(2, lll_size), n)

    set_num_cores(cores)
    set_debug_flag(1 if debug else 0)

    tracers = {}
    if verbose:
        def trace_print(_, prof, note):
            log_str = '.'
            if note[0].startswith('BKZ'):
                beta, tour, ntours, touridx = note[1]
                log_str = (f"\nBKZ(β:{beta:3d},t:{tour + 1:2d}/{ntours:2d}, o:{touridx:4d}): "
                           f"slope={slope(prof):.6f}, rhf={rhf(prof):.6f}")
            print(log_str, end="", file=stderr, flush=True)
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
        import matplotlib.pyplot as plt
        from matplotlib.animation import ArtistAnimation, PillowWriter
    if has_animation:
        fig, ax = plt.subplots()
        ax.set(xlim=[0, n])
        artists = []

        def trace_anim(_, prof, __):
            artists.append(ax.plot(range(n), prof, color="blue"))

        tracers['a'] = trace_anim

    B = B.copy()  # Do not modify B in-place, but work with a copy.
    U = np.identity(n, dtype=np.int64)
    U_seysen = np.identity(n, dtype=np.int64)
    beta = kwds.get("beta")
    hkz_use = kwds.get("hkz_use")
    hkz_prog = kwds.get("hkz_prog")
    pump_and_jump = kwds.get("pump_and_jump")
    time_start = perf_counter_ns()
    try:
        if not beta:
            lll_reduce_gpu(B, U, U_seysen, lll_size, delta, depth, tprof, tracers, debug, use_seysen)
        else:
            # Parse BKZ parameters:
            bkz_tours = kwds.get("bkz_tours") or 1
            bkz_size = kwds.get("bkz_size") or lll_size
            bkz_prog = kwds.get("bkz_prog") or beta

            # Progressive-BKZ: start running BKZ-beta' for some `beta' >= 40`,
            # then increase the blocksize beta' by `bkz_prog` and run BKZ-beta' again,
            # and repeat this until `beta' = beta`.
            betas = range(40 + ((beta - 40) % bkz_prog), beta + 1, bkz_prog)

            switch_over = 64

            # In the literature on BKZ, it is usual to run LLL before calling the SVP oracle in BKZ.
            # However, it is actually better to preprocess the basis with 4-deep-LLL instead of LLL,
            # before calling the SVP oracle.
            for beta_ in betas:
                if hkz_use: 
                    if hkz_prog:
                        if beta_ <= switch_over:
                            bkz_reduce(B, U, U_seysen, lll_size, delta, 4, beta_,
                           bkz_tours if beta_ == beta else 1, bkz_size,
                           tprof, tracers, debug, use_seysen)
                        else:
                            hkz_reduce(B, U, U_seysen, lll_size, delta, 4, beta_,
                           bkz_tours if beta_ == beta else 1, bkz_size,
                           tprof, tracers, debug, use_seysen, pump_and_jump)


                    else:
                        hkz_reduce(B, U, U_seysen, lll_size, delta, 4, beta_,
                           bkz_tours if beta_ == beta else 1, bkz_size,
                           tprof, tracers, debug, use_seysen, pump_and_jump)
                else:
                    bkz_reduce_gpu(B, U, U_seysen, lll_size, delta, 4, beta_,
                           bkz_tours if beta_ == beta else 1, bkz_size,
                           tprof, tracers, debug, use_seysen)
    except KeyboardInterrupt:
        pass  # When interrupted, give the partially reduced basis.
    time_end = perf_counter_ns()
    print(f"\nTotal time: {(time_end - time_start) * 10**-9:.3f}s", file=stderr)
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
    print(tprof, file=stderr)
    return U, B, tprof