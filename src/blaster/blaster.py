"""
BLASter lattice reduction: LLL with QR decomposition, Seysen's reduction, and
segments, in which lattice reduction is done in parallel.
"""
from functools import partial
from sys import stderr
from time import perf_counter_ns

import numpy as np
import cupy as cp

# Local imports
from blaster_core import \
    set_debug_flag, set_num_cores, block_lll, block_deep_lll, block_bkz, ZZ_right_matmul ,get_R_sub_HKZ, apply_U_HKZ, block_lll_gpu, block_deep_lll_gpu, block_bkz_gpu
from .size_reduction import is_lll_reduced, is_weakly_lll_reduced, size_reduce, seysen_reduce

from .size_reduction_gpu import is_weakly_lll_reduced_gpu, seysen_reduce_gpu

from .stats import get_profile, rhf, slope, potential, get_profile_gpu
from .lattice_io import write_lattice
from fpylll.util import gaussian_heuristic

from .hkz import hkz_kernel

# from size_reduction import is_lll_reduced, is_weakly_lll_reduced, size_reduce, seysen_reduce

# from size_reduction_gpu import is_weakly_lll_reduced_gpu, seysen_reduce_gpu

# from stats import get_profile, rhf, slope, potential, get_profile_gpu
# from lattice_io import write_lattice

# from hkz import hkz_kernel

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


# from lll_gpu import lll_reduction_gpu
def is_seysen_reduced(R, tol=1e-8):
    #R11⁻1 R12 max <= 1/2
    n = R.shape[0]
    k = n // 2

    # Extraction des quatre blocs
    R11 = R[:k,   :k]    # lignes 0 à k-1, colonnes 0 à k-1
    R12 = R[:k,   k:]    # lignes 0 à k-1, colonnes k à n-1
    #check that is seysen reduce
    R11inv = np.linalg.inv(R11)
    result = R11inv @ R12
    print(np.max(np.abs(result)))
    return np.max(np.abs(result)) < 1/2



def lll_reduce_gpu(B, U, U_seysen, lll_size, delta, depth,
               tprof, tracers, debug, use_seysen, R_cpu=None):
    """
    Perform BLASter's lattice reduction on basis B, and keep track of the transformation in U.
    If `depth` is supplied, use deep insertions up to depth `depth`.
    """
    n, is_reduced, offset = B.shape[1], 0, 0


    host_flag = cp.cuda.alloc_pinned_memory(1)
    #is like a memset because if not it can don't iterate
    mv = memoryview(host_flag)
    mv[0] = 0
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

    if R_cpu is None:
        n, m = B_gpu.shape
        pinned_host_arr = cp.cuda.alloc_pinned_memory(n* m * 8) #memory needed

        # On wrappe ça en numpy pour manipuler :
        R_cpu = np.frombuffer(pinned_host_arr, dtype=np.float64, count=n*m).reshape((n, m))

    logging = True
    while not is_reduced:
        # — Step 1: QR on GPU
        t1 = perf_counter_ns()
        R_gpu = cp.linalg.qr(B_gpu, mode='r')
        # — Step 2: small‐block LLL on GPU (further work)

        # CPU fallback
        # 1) transfer back to host
        cp.asnumpy(R_gpu, out=R_cpu) # synchro CPU-GPU so this is the time to compute also the other stuff

        t2 = perf_counter_ns()
        offset = lll_size//2 if offset==0 else 0
        if use_gpu_lll:  # once you have a GPU version
            raise "Not implemented yet"
        else:
            
            # 2) run existing CPU version
            U_sub = red_fn(R_cpu, delta, offset, lll_size) # num_blocks, block_size**2

            # #copy to the GPU
            U_sub_gpu = cp.asarray(U_sub)

            # Calcul du nombre de blocs
            num_blocks = int((n - offset + lll_size - 1) // lll_size)
            for block_id in range(num_blocks): # this loop is faster than a batch, tested for n <= 4096
                i = offset + lll_size * block_id
                w = min(n - i, lll_size)
                block_vals = U_sub_gpu[block_id, : w*w].reshape(w, w)
                # Au lieu de U_sub_total, on réécrit juste les colonnes concernées :
                U_gpu[:, i : i + w] = U_gpu[:, i : i + w] @ block_vals
                B_gpu[:, i : i + w] = B_gpu[:, i : i + w] @ block_vals


            # batched implementation tested    
                # n = U_gpu.shape[0]
                # w = lll_size
                # full_blocks = (n - offset) // w
                # last_w = (n - offset) - full_blocks * w

                # if full_blocks > 0:
                #     block_vals_full = (
                #         U_sub_gpu[:full_blocks, : w*w]
                #         .reshape(full_blocks, w, w)
                #     )

                # if full_blocks > 0:
                #     # Slice of full_blocks
                #     U_slice = U_gpu[:, offset : offset + full_blocks * w]        # (n, full_blocks*w)
                #     U_blocks = U_slice.reshape(n, full_blocks, w)                # (n, full_blocks, w)
                #     U_blocks_t = U_blocks.transpose(1, 0, 2)                     # (full_blocks, n, w)
                #     out_full = cp.matmul(U_blocks_t, block_vals_full)           # (full_blocks, n, w)
                #     U_gpu[:, offset : offset + full_blocks * w] = (
                #         out_full.transpose(1, 0, 2).reshape(n, full_blocks * w)
                #     )

                # # fallback if len of the last block is < w
                # if last_w > 0:
                #     blk_id = full_blocks
                #     i = offset + full_blocks * w
                #     block_vals_last = (
                #         U_sub_gpu[blk_id, : last_w*last_w]
                #         .reshape(last_w, last_w)
                #     )
                #     U_gpu[:, i : i + last_w] = U_gpu[:, i : i + last_w] @ block_vals_last

                # if full_blocks > 0:
                #     B_slice = B_gpu[:, offset : offset + full_blocks * w]       # (n, full_blocks*w)
                #     B_blocks = B_slice.reshape(n, full_blocks, w)               # (n, full_blocks, w)
                #     B_blocks_t = B_blocks.transpose(1, 0, 2)                     # (full_blocks, n, w)

                #     outB_full = cp.matmul(B_blocks_t, block_vals_full)          # (full_blocks, n, w)
                #     B_gpu[:, offset : offset + full_blocks * w] = (
                #         outB_full.transpose(1, 0, 2).reshape(n, full_blocks * w)
                #     )

                # if last_w > 0:
                #     B_gpu[:, i : i + last_w] = B_gpu[:, i : i + last_w] @ block_vals_last

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
        # assert is_seysen_reduced(R_gpu)
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

    logging = True

    is_B_gpu = isinstance(B, cp.ndarray)
    is_U_gpu = isinstance(U, cp.ndarray)
    is_U_s_gpu = isinstance(U_seysen, cp.ndarray)

    # 2) …and lift any host arrays up to the device once
    B_gpu        = B        if is_B_gpu   else cp.asarray(B)
    U_gpu        = U        if is_U_gpu   else cp.asarray(U)
    U_s_gpu      = U_seysen if is_U_s_gpu else cp.asarray(U_seysen)
    n, m = B_gpu.shape
    pinned_host_arr = cp.cuda.alloc_pinned_memory(n* m * 8) #memory needed

        # On wrappe ça en numpy pour manipuler :
    R_cpu = np.frombuffer(pinned_host_arr, dtype=np.float64, count=n*m).reshape((n, m))

    lll_reduce_gpu(B_gpu, U_gpu, U_s_gpu, lll_size, delta, depth, tprof, tracers, debug, use_seysen, R_cpu)
    while tours_done < bkz_tours:
        # Step 1: QR-decompose B, and only store the upper-triangular matrix R.
        t1 = perf_counter_ns()
        # R = np.linalg.qr(B, mode='r')
        R_gpu = cp.linalg.qr(B_gpu, mode='r')

        # Step 2: Call BKZ concurrently on small blocks!
        t2 = perf_counter_ns()
        print("(BKZ) we are at :", cur_front)
        # norm_before = abs(R[cur_front, cur_front])

        cp.asnumpy(R_gpu, out= R_cpu)
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
        if logging:
            prof = get_profile_gpu(R_gpu, True).get()
            note = (f"BKZ-{beta}", (bkz_size, tours_done, bkz_tours, cur_front))
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
        lll_reduce_gpu(B_gpu, U_gpu, U_s_gpu, lll_size, delta, depth, tprof, tracers, debug, use_seysen, R_cpu)
    if not is_U_gpu:
        U[:] = cp.asnumpy(U_gpu)
    if not is_B_gpu:
        B[:] = cp.asnumpy(B_gpu)
    if not is_U_s_gpu:
        U_seysen[:] = cp.asnumpy(U_s_gpu)
               

def get_R_sub_HKZ_gpu(R_gpu: cp.ndarray, cur_front: int, w: int) -> cp.ndarray:
    return R_gpu[cur_front:cur_front+w, cur_front:cur_front+w].copy()


def apply_U_HKZ_gpu(B_red_gpu, U_gpu, U_sub, cur_front, w):
    # 1) Convertir U_sub CPU → GPU, et forcer même dtype que U_gpu
    U_sub_gpu = cp.asarray(U_sub, dtype=U_gpu.dtype)

    # 2) Bornes du bloc
    i = cur_front
    j = cur_front + w

    # 3) Vues-slices (potentiellement non‑contigues)
    U_block_view = U_gpu   [:, i:j]   # shape (n, w)
    B_block_view = B_red_gpu[:, i:j]  # shape (m, w)

    # 4) Copie pour obtenir un buffer C‑contigu
    U_block = U_block_view.copy()
    B_block = B_block_view.copy()

    # 5) Produit matriciel hors‑place
    U_res = U_block @ U_sub_gpu
    B_res = B_block @ U_sub_gpu

    # 6) Réécriture dans U_gpu et B_red_gpu via slice (qui accepte un array contigu)
    U_gpu   [:, i:j] = U_res
    B_red_gpu[:, i:j] = B_res

#need rework output of this is completely wrong
# def hkz_reduce_gpu(B, U, U_seysen, lll_size, delta, depth,
#                beta, bkz_tours, block_size, tprof, tracers, debug, use_seysen, pump_and_jump, svp_call = False, target_norm=None, goal_margin=1.5):
    
#     # BKZ parameters:
#     n, tours_done, cur_front = B.shape[1], 0, 0
#     if svp_call:
#         cur_front = n - beta    # from Erdem Alkim, Léo Ducas, Thomas Pöppelmann, and Peter Schwabe. Post-quantum key exchange a new hope
#         #and from G6K
#         proj_target_norm =  target_norm * (n - beta)/n
#         proj_gh             = gaussian_heuristic([g6k.M.get_r(i, i)
#                              for i in range(llb, d)])
#         proj_target_norm_slack = min(goal_margin * proj_target_norm,
#                              0.98 * proj_gh)

#     logging = True

#     is_B_gpu = isinstance(B, cp.ndarray)
#     is_U_gpu = isinstance(U, cp.ndarray)
#     is_U_s_gpu = isinstance(U_seysen, cp.ndarray)

#     # 2) …and lift any host arrays up to the device once
#     B_gpu        = B        if is_B_gpu   else cp.asarray(B)
#     U_gpu        = U        if is_U_gpu   else cp.asarray(U)
#     U_s_gpu      = U_seysen if is_U_s_gpu else cp.asarray(U_seysen)
#     n, m = B_gpu.shape
#     pinned_host_arr = cp.cuda.alloc_pinned_memory(n* m * 8) #memory needed

#         # On wrappe ça en numpy pour manipuler :
#     R_cpu = np.frombuffer(pinned_host_arr, dtype=np.float64, count=n*m).reshape((n, m))

#     lll_reduce_gpu(B_gpu, U_gpu, U_s_gpu, lll_size, delta, depth, tprof, tracers, debug, use_seysen, R_cpu)
#     if block_size < beta:
#         block_size = beta
#     while tours_done < bkz_tours:
#         # Step 1: QR-decompose B, and only store the upper-triangular matrix R.
#         t1 = perf_counter_ns()
#         # R = np.linalg.qr(B, mode='r')
#         PRE_B_GPU = B_gpu.copy()
#         R_gpu = cp.linalg.qr(B_gpu, mode='r')

#         # Step 2: Call BKZ concurrently on small blocks!
#         t2 = perf_counter_ns()
#         print("(HKZ) we are at :", cur_front)
#         # norm_before = abs(R[cur_front, cur_front])

#         w = block_size if (n - cur_front) >= block_size else (n - cur_front)

#         B_cpu = cp.asnumpy(B_gpu)
#         U_cpu = cp.asnumpy(U_gpu)
#         cp.asnumpy(R_gpu, out=R_cpu)
#         #error here with all in gpu, prefer call back all matrix on CPU

#         R_sub = get_R_sub_HKZ(R_cpu, cur_front, w)
#         U_sub = hkz_kernel(R_sub, w, beta, pump_and_jump)
#         apply_U_HKZ(B_cpu, U_cpu, U_sub, cur_front, w)

#         B_gpu = cp.asarray(B_cpu)
#         U_gpu = cp.asarray(U_cpu)

#         print("B_gpu unchanged:", bool((B_gpu == PRE_B_GPU).all()))
#         w = U_sub.shape[0]
#         I_gpu = cp.eye(w, dtype=U_sub.dtype)

#         print(
#             "U_sub is identity:",
#             bool(cp.allclose(U_sub, I_gpu))
#         )
                

#         # Step 3: QR-decompose again because BKZ "destroys" the QR decomposition.
#         # Note: it does not destroy the bxb blocks, but everything above these: yes!
#         t3 = perf_counter_ns()
#         R_gpu = cp.linalg.qr(B_gpu, mode='r')
#         # print(abs(R[cur_front, cur_front]), norm_before)
#         # assert abs(R[cur_front, cur_front]) <= norm_before
#         # Step 4: Seysen reduce or size reduce the upper-triangular matrix R.
#         t4 = perf_counter_ns()
#         if use_seysen:
#             seysen_reduce_gpu(R_gpu, U_s_gpu)
#         else:
#             raise "Error not Implemented"

#         # Step 5: Update B and U with transformation from Seysen's reduction.
#         t5 = perf_counter_ns()
#         U_gpu = U_gpu @ U_s_gpu
#         B_gpu = B_gpu @ U_s_gpu

#         t6 = perf_counter_ns()

#         tprof.tick(t2 - t1 + t4 - t3, 0, t3 - t2, t5 - t4, t6 - t5)

#         # After time measurement:
#         if logging:
#             prof = get_profile_gpu(R_gpu, True).get()
#             note = (f"HKZ-{beta}", (block_size, tours_done, bkz_tours, cur_front))
#             for tracer in tracers.values():
#                 tracer(tprof.num_iterations, prof, note)
#         # After printing: update the current location of the 'reduction front'
#         if cur_front + beta >= n:
#             # HKZ-reduction was performed at the end, which is the end of a tour.
#             cur_front = 0
#             tours_done += 1
#         else:
#             cur_front += (block_size - beta + 1)
#             #cur_front += 1

#         # Perform a final LLL reduction at the end
#         lll_reduce_gpu(B_gpu, U_gpu, U_s_gpu, lll_size, delta, depth, tprof, tracers, debug, use_seysen, R_cpu)
#     if not is_U_gpu:
#         U[:] = cp.asnumpy(U_gpu)
#     if not is_B_gpu:
#         B[:] = cp.asnumpy(B_gpu)
#     if not is_U_s_gpu:
#         U_seysen[:] = cp.asnumpy(U_s_gpu)

def hkz_reduce(B, U, U_seysen, lll_size, delta, depth,
               beta, bkz_tours, block_size, tprof, tracers, debug, use_seysen, pump_and_jump, svp_call=False, target_norm=None, goal_margin=1.5):
    """
    Perform BLASter's BKZ reduction on basis B, and keep track of the transformation in U.
    If `depth` is supplied, BLASter's deep-LLL is called in between calls of the SVP oracle.
    Otherwise BLASter's LLL is run.
    """
    # BKZ parameters:
    n, tours_done, cur_front = B.shape[1], 0, 0
    if svp_call:
            if not target_norm:
                raise("You need to set a target norm")
            cur_front = n - beta

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

        if svp_call:
            Ug6k = hkz_kernel(B, w, beta, pump_and_jump, target_norm)
            ZZ_right_matmul(U, Ug6k)
            ZZ_right_matmul(B, Ug6k)
        else:
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
        print('\nProfile sortie de SVP = [' + ' '.join([f'{x:.2f}' for x in prof]) + ']\n'
              f'RHF = {rhf(prof):.5f}^n, slope = {slope(prof):.6f}, '
              f'∥b_1∥ = {2.0**prof[0]:.1f}', file=stderr)
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


# def hybrid_block_reduction(
#     B_gpu: cp.ndarray,
#     U_gpu: cp.ndarray,
#     cur_front: int,
#     block_size: int,
#     beta: int,
#     hkz_kernel,
#     apply_U_HKZ_gpu,
#     nearest_plane,
#     pump_and_jump: bool = False
# ) -> None:
#     """
#     Apply a hybrid BKZ-SVP + Babai nearest-plane reduction on a GPU.

#     Steps:
#     1. Extract upper-triangular R from B_gpu via QR.
#     2. Select sub-block of width w = min(block_size, n - cur_front).
#     3. Run SVP kernel (hkz_kernel) on R_sub to get U_sub (beta-dimensional SVP).
#     4. Apply U_sub to B_gpu and U_gpu for that block.
#     5. Build the target vector t in R^n from SVP coefficients.
#     6. Perform Babai nearest-plane on full R to lift the solution.
#     7. Update B_gpu and U_gpu with the Babai transformation.
#     """
#     n = B_gpu.shape[1]
#     # 1. QR decomposition of B
#     R_gpu = cp.linalg.qr(B_gpu, mode='r')

#     # 2. Sub-block selection
#     w = block_size if (n - cur_front) >= block_size else (n - cur_front)
#     R_sub = get_R_sub_HKZ_gpu(R_gpu, cur_front, w)         # GPU view
#     R_sub_cpu = cp.asnumpy(R_sub)                          # to CPU for SVP kernel

#     # 3. SVP on sub-block
#     U_sub = hkz_kernel(R_sub_cpu, w, beta, pump_and_jump)

#     # 4. Apply SVP transformation
#     apply_U_HKZ_gpu(B_gpu, U_gpu, U_sub, cur_front, w)

#     # 5. Build target vector t in CPU
#     # Extract SVP coefficients from U_sub: assume U_sub is w x w transformation
#     coeffs_eta = U_sub[:, 0][:beta]                       # first column gives combination
#     c = np.zeros(n, dtype=int)
#     c[cur_front:cur_front + beta] = coeffs_eta
#     B_cpu = cp.asnumpy(B_gpu)
#     t = B_cpu @ c

#     # 6. Nearest-plane on full R
#     # Prepare T and U_babai
#     T = t.reshape(n, 1)
#     U_babai = np.zeros((n, 1), dtype=int)
#     R_cpu = cp.asnumpy(R_gpu)
#     nearest_plane(R_cpu, T, U_babai)

#     # 7. Update GPU bases with Babai transform
#     # Transfer U_babai to GPU as full transformation matrix
#     # Here we treat U_babai as the extra columns on identity
#     U_full = np.eye(n, dtype=int)
#     U_full[:, 0] = U_babai.flatten()
#     U_full_gpu = cp.asarray(U_full)
#     # Apply to global U and B
#     U_gpu[...] = U_gpu @ U_full_gpu
#     B_gpu[...] = B_gpu @ U_full_gpu


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
    svp_call = kwds.get("svp_call")
    target_norm = kwds.get("target")
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

            if svp_call:
                print("svp call")
                hkz_reduce(B, U, U_seysen, lll_size, delta, 4, beta,
                           bkz_tours, bkz_size,
                           tprof, tracers, debug, use_seysen, pump_and_jump, svp_call, target_norm)
            else:
                # In the literature on BKZ, it is usual to run LLL before calling the SVP oracle in BKZ.
                # However, it is actually better to preprocess the basis with 4-deep-LLL instead of LLL,
                # before calling the SVP oracle.
                # moreover we can call DeepLLL-30 before everything so it's will be really more faster
                # lll_reduce_gpu(B, U, U_seysen, lll_size, delta, 30, tprof, tracers, debug, use_seysen)
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
                            bkz_reduce(B, U, U_seysen, lll_size, delta, 4, beta_,
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
    # prof = get_profile(B)
    # print('\nProfile = [' + ' '.join([f'{x:.2f}' for x in prof]) + ']\n'
    #           f'RHF = {rhf(prof):.5f}^n, slope = {slope(prof):.6f}, '
    #           f'∥b_1∥ = {2.0**prof[0]:.1f}', file=stderr)
    return U, B, tprof
