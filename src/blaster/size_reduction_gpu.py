"""
Utility functions for calling Babai's nearest plane algorithm, size-reducing a basis or
Seysen-reducing a basis.

In comments, the old recursive functions are kept for clarity.
"""
from functools import cache
import cupy as cp
from cupyx.scipy.linalg import solve_triangular
# Local imports


def is_weakly_lll_reduced_gpu(R, host_flag, delta=0.99):
    """
    Fully GPU-side check for the Weak-LLL condition:
      ∀ pos: ||b_{pos+1}||^2 > δ · ||b_pos||^2
    where b_pos = (R[pos,pos], 0) and b_{pos+1} = (〈R[pos,pos+1]〉_u, R[pos+1,pos+1]).
    Returns a Python bool, but only synchronizes once at the very end.
    """
    n = R.shape[0]
    # positions 0 .. n-2
    i = cp.arange(n - 1)

    # diagonal and super-diagonal entries
    u = cp.abs(R[i,     i    ])   # shape (n-1,)
    v =        R[i,     i + 1]   # shape (n-1,)
    w =        R[i + 1, i + 1]   # shape (n-1,)

    # compute centered v_mod = (v mod u) in [−u/2, +u/2)
    v_mod = ((v + u/2) % u) - u/2

    # LLL condition: v_mod**2 + w**2 > δ * u**2
    ok = (v_mod**2 + w**2) > (delta * u**2)

    # cp.all returns a 0-d GPU array; .item() brings back one host bool
    ok_scalar = cp.all(ok)   # GPU-side scalar
    ptr_dev = ok_scalar.data.ptr
    ptr_host = host_flag.ptr  # pointeur hôte
    stream = cp.cuda.Stream.null
    cp.cuda.runtime.memcpyAsync(
        ptr_host, ptr_dev,
        1,  # 1 byte – cupy bool
        cp.cuda.runtime.memcpyDeviceToHost,
        stream.ptr
    )


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()

def __reduction_ranges_pow2(n: int):
    """
    Intervalles (virtuellement) sur M=2^ceil(log2 n), puis clipping à [0,n).
    - ordre topologique garanti: tailles croissantes (s = 4,8,...,M)
    - à taille fixée, on regroupe/intériorise pour maximiser (h,w) identiques
    Retourne (base_cases, triples).
    """
    M = _next_pow2(n)

    # base cases = feuilles s==2 de la décomposition sur [0,M), puis clip
    base_cases = []
    for l in range(0, M, 2):     # intervalles [l,l+2)
        i = min(l, n)
        j = min(l+1, n)
        if j < n:                # garde uniquement i s.t. i+1 existe dans [0,n)
            base_cases.append(i)

    # niveaux s = 4,8,...,M
    result = []
    for e in range(2, M.bit_length()+1):
        s = 1 << e
        level = []
        for l in range(0, M, s):
            m = l + (s >> 1)
            r = l + s
            # clip à [0,n]
            li, mi, ri = min(l, n), min(m, n), min(r, n)
            # jette les triples vides/dégénérés
            if li < mi < ri:
                level.append((li, mi, ri))

        # maximise les runs homogènes dans le niveau
        # tri: intérieurs d'abord (k!=n), puis par (h, w)
        level.sort(key=lambda t: (t[2] == n, t[1]-t[0], t[2]-t[1]))
        result.extend(level)

    # (rare) dédup si clipping crée des doublons
    seen, uniq = set(), []
    for t in result:
        if t not in seen:
            uniq.append(t); seen.add(t)

    return base_cases, uniq


import cupy as cp

__reduction_cache = {}
_rows_h_cache = {}
_cols_w_cache = {}

def dynamic_batches(ranges, N, min_batch=8):
    """
    Découpe `ranges` en batches dynamiques:
    - s'arrête et démarre un nouveau batch dès qu'un triplet a k == N
    - étend le batch tant que la largeur (k-j) n'excède pas la largeur max actuelle
      ou que le batch n'a pas atteint min_batch
    - autorise un batch plus petit que min_batch uniquement si la coupure est due à k == N

    Retourne une liste de listes :
      [batch_size, current_ranges, max_h, max_w]
    où max_h = max(j - i) et max_w = max(k - j) sur le batch.
    """
    m = len(ranges)
    offset = 0
    batches = []

    while offset < m:
        # init du batch
        i0, j0, k0 = ranges[offset]
        max_w = k0 - j0
        max_h = j0 - i0
        batch_size = 1
        hit_N = False

        # essayer d'étendre le batch
        while offset + batch_size < m:
            i1, j1, k1 = ranges[offset + batch_size]
            w1 = k1 - j1
            h1 = j1 - i1

            # coupure forcée sur k == N
            if k1 == N:
                hit_N = True
                break

            # si ca change, on stoppe
            if w1 != max_w or h1 != max_h:
                break
            batch_size += 1

        # extraction du batch
        current_ranges = ranges[offset:offset + batch_size]
        batches.append([batch_size, current_ranges, max_h, max_w])

        # avancer
        offset += batch_size

    return batches

def seysen_reduce_gpu(R_gpu, U_gpu):
    """
    GPU version of Seysen's reduction on an upper-triangular matrix R_gpu,
    tracking the transformation in U_gpu. Both inputs are modified in place.

    :param R_gpu: cp.ndarray, upper-triangular matrix to reduce (float dtype)
    :param U_gpu: cp.ndarray, integer transformation matrix (upper-triangular with 1s on diag)
    """
    n = R_gpu.shape[0]
    dev_id = int(R_gpu.device.id)

    key = (dev_id, n)

    #param ici
    min_batch = 4
    min_batch_low = 8
    area_limit = 64*64

    # Use cached ranges if available
    if key not in __reduction_cache:

        base_cases, ranges = __reduction_ranges_pow2(n)

        i_arr = cp.array(base_cases, dtype=cp.int32)
        batches = dynamic_batches(ranges,n)

        ranges_np_full = cp.asarray(ranges, dtype=cp.int32)

        __reduction_cache[key] = (base_cases, ranges, i_arr, ranges_np_full, batches)
    else:
        base_cases, ranges, i_arr, ranges_np_full, batches = __reduction_cache[key]

    if base_cases:
        # gather diagonals and super‐diagonals
        u = R_gpu[i_arr, i_arr]
        v = R_gpu[i_arr, i_arr + 1]
        t = -cp.rint(v / u)
        # update U and R in‐place
        U_gpu[i_arr, i_arr + 1] = t.astype(U_gpu.dtype)
        R_gpu[i_arr, i_arr + 1] += u * t

    # # 2) Main reduction loops
    Uf_gpu = U_gpu.astype(R_gpu.dtype, copy=True)  # float‐work copy (only once)

    m = len(ranges)
    level = 0
    offset = 0

    for batch_size, current_ranges, max_h, max_w  in batches:
        if batch_size >= min_batch and not (batch_size <= min_batch_low and (max_h * max_w) < area_limit):
            ranges_np = ranges_np_full[offset : offset + batch_size]
            i_arr, j_arr, k_arr = ranges_np[:,0], ranges_np[:,1], ranges_np[:,2]

            # 2) préparer caches
            rows_key = (dev_id, max_h)
            cols_key = (dev_id, max_w)
            with R_gpu.device:
                rows_h = _rows_h_cache.setdefault(rows_key, cp.arange(max_h, dtype=cp.int32))
                cols_w = _cols_w_cache.setdefault(cols_key, cp.arange(max_w, dtype=cp.int32))

            # 3) construire I, J
            I = i_arr[:, None] + rows_h[None, :]   # (batch_size, max_h)
            J = j_arr[:, None] + cols_w[None, :]   # (batch_size, max_w)

            # 4) gather sous-blocs
            R11 = R_gpu[I[:,:,None], I[:,None,:]]      # (b, h, h)
            R12 = R_gpu[I[:,:,None], J[:,None,:]]      # (b, h, w)
            U22 = Uf_gpu[J[:,:,None], J[:,None,:]]     # (b, w, w)
            # 5) GEMM + solve -> U12 (b, h, w)
            S12 = cp.matmul(R12, U22)
            U12 = cp.rint(-cp.linalg.solve(R11, S12))

            # 6) mise à jour des sous-blocs
            Uf_block    = Uf_gpu[I[:,:,None], I[:,None,:]]  # (b, h, h)
            R12_update  = cp.matmul(R11, U12)               # (b, h, w)
            Uf12_update = cp.matmul(Uf_block, U12)          # (b, h, w)
            I_idx = I[:,:,None].repeat(max_w, axis=2)   # (b, h, w)
            J_idx = J[:,None,:].repeat(max_h, axis=1)   # (b, h, w)
            I_flat = I_idx.reshape(-1)
            J_flat = J_idx.reshape(-1)
            # 8) scatter-update
            R_gpu[I_flat, J_flat]  = (S12 + R12_update).reshape(-1)
            Uf_gpu[I_flat, J_flat] = Uf12_update.reshape(-1)
        else:
            for (i, j, k) in current_ranges:
                # S12' = R[i:j, j:k] @ U[j:k, j:k]
                S12p = R_gpu[i:j, j:k].dot(Uf_gpu[j:k, j:k])

                # U12' = round(-solve(R[i:j, i:j], S12'))
                S11 = R_gpu[i:j, i:j]
                U12p = cp.rint(-solve_triangular(S11, S12p))

                # R[i:j, j:k] = S12' + S11 @ U12'
                R_gpu[i:j, j:k] = S12p + S11.dot(U12p)

                # U[i:j, j:k] = U[i:j, i:j] @ U12'
                Uf_gpu[i:j, j:k] = (Uf_gpu[i:j, i:j].dot(U12p))
        offset+= batch_size
    U_gpu[:, :] = cp.rint(Uf_gpu).astype(U_gpu.dtype)



import gc
def _pool_report(tag=""):
    free, total = cp.cuda.runtime.memGetInfo()
    used = total - free
    mp = cp.get_default_memory_pool()
    pp = cp.get_default_pinned_memory_pool()
    print(f"[{tag}] used={used/1e9:.2f}GB | pool_used={mp.used_bytes()/1e9:.2f}GB | pool_held={mp.total_bytes()/1e9:.2f}GB")

def clear_internal_caches(trim_pools=True, all_devices=True, verbose=False):
    global __reduction_cache, _rows_h_cache, _cols_w_cache

    if verbose: _pool_report("before")

    # 1) Drop refs Python
    __reduction_cache.clear()
    _rows_h_cache.clear()
    _cols_w_cache.clear()

    # 2) GC + sync pour être sûr que rien n'est encore en vol
    gc.collect()

    if all_devices:
        ndev = cp.cuda.runtime.getDeviceCount()
        for dev in range(ndev):
            with cp.cuda.Device(dev):
                cp.cuda.runtime.deviceSynchronize()
                if trim_pools:
                    cp.get_default_memory_pool().free_all_blocks()
    else:
        cp.cuda.runtime.deviceSynchronize()
        if trim_pools:
            cp.get_default_memory_pool().free_all_blocks()
    if trim_pools:
        cp.get_default_pinned_memory_pool().free_all_blocks()

    if verbose: _pool_report("after")


import numpy as np

import numpy as np
import cupy as cp

def babai_last_gpu_batched(b_last_batch_gpu, Q_gpu, R_gpu, eps=1e-12):
    """
    Batched Babai on a batch of last columns
    """
    # t = Q^T b, TRSV batched, u = -round(c), w = R u
    T = Q_gpu.T @ b_last_batch_gpu
    #solve triangular batched only with prerelease cupy (needed to by tried)
    
    # C = solve_triangular(R_gpu, T, lower=False, trans='N',
    #                      unit_diagonal=False, check_finite=False)
    C = cp.linalg.solve(R_gpu, T)
    U = -cp.rint(C)
    W = R_gpu @ U
    bprime_batch = b_last_batch_gpu + Q_gpu @ W
    return bprime_batch