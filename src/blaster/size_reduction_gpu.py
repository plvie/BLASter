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
    v_mod = v - cp.rint(v / u) * u

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



@cache
def __reduction_ranges(n):
    """
    Return list of ranges that needs to be reduced.

    More generally, it returns, without using recursion, the list that would be
    the output of the following Python program:

    <<<BEGIN CODE>>>
    def rec_range(n):
        bc, res = [], []
        def F(l, r):
            if l == r:
                return
            if l + 1 == r:
                bc.append(l)
            else:
                m = (l + r) // 2
                F(l, m)
                F(m, r)
                res.append((l, m, r))
        return F(0, n)
    <<<END CODE>>>

    :param n: the length of the array that requires reduction
    :return: pair containing `the base_cases` and `result`.
             `base_cases` is a list of indices `i` such that:
                `i + 1` needs to be reduced w.r.t. `i`.
             `result` is a list of triples `(i, j, k)` such that:
                `[j:k)` needs to be reduced w.r.t. `[i:j)`.
             The guarantee is that for any 0 <= i < j < n:
             1) `i in base_cases && j = i + 1`,
             OR
             2) there is a triple (u, v, w) such that `i in [u, v)` and `j in [v, w)`.
    """
    bit_shift, parts, result, base_cases = 1, 1, [], []
    while parts < n:
        left_bound, left_idx = 0, 0
        for i in range(1, parts + 1):
            right_bound = left_bound + 2 * n

            mid_idx = (left_bound + n) >> bit_shift
            right_idx = right_bound >> bit_shift

            if right_idx > left_idx + 1:
                # Only consider nontrivial intervals
                if right_idx == left_idx + 2:
                    # Return length 2 intervals separately to unroll base case.
                    base_cases.append(left_idx)
                else:
                    # Properly sized interval:
                    result.append((left_idx, mid_idx, right_idx))
            left_bound, left_idx = right_bound, right_idx
        parts *= 2
        bit_shift += 1
    return base_cases, list(reversed(result))


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


__workspace_cache = {}  # key: (dtype.str, b, h, w) -> dict(S, U, Z)

def _get_workspace(dtype, b, h, w):
    """
    Réutilise des buffers 3D (b,h,w) pour éviter les allocs répétées.
    S: buffer polyvalent (S12 puis X=R11^{-1}S)
    U: U12 (arrondi) puis delta
    Z: Uf12 (résultat GEMM)
    """
    dt = cp.dtype(dtype)
    key = (dt.str, int(b), int(h), int(w))
    ws = __workspace_cache.get(key)
    if ws is None:
        ws = {
            "S": cp.empty((b, h, w), dtype=dt, order="C"),
            "U": cp.empty((b, h, w), dtype=dt, order="C"),
            "Z": cp.empty((b, h, w), dtype=dt, order="C"),
        }
        __workspace_cache[key] = ws
    return ws["S"], ws["U"], ws["Z"]

def seysen_reduce_gpu(R_gpu, U_gpu):
    """
    GPU version of Seysen's reduction on an upper-triangular matrix R_gpu,
    tracking the transformation in U_gpu. Both inputs are modified in place.

    :param R_gpu: cp.ndarray, upper-triangular matrix to reduce (float dtype)
    :param U_gpu: cp.ndarray, integer transformation matrix (upper-triangular with 1s on diag)
    """
    n = R_gpu.shape[0]
    dtype = R_gpu.dtype

    #param ici
    min_batch = 8
    area_batch = 32*32

    # Use cached ranges if available
    if n not in __reduction_cache:

        base_cases, ranges = __reduction_ranges(n)

        i_arr = cp.array(base_cases, dtype=cp.int32)
        batches = dynamic_batches(ranges,n)

        ranges_np_full = cp.asarray(ranges, dtype=cp.int32)

        __reduction_cache[n] = (base_cases, ranges, i_arr, ranges_np_full, batches)
    else:
        base_cases, ranges, i_arr, ranges_np_full, batches = __reduction_cache[n]

    if base_cases:
        # gather diagonals and super‐diagonals
        u = R_gpu[i_arr, i_arr]
        v = R_gpu[i_arr, i_arr + 1]
        t = -cp.rint(v / u)
        # update U and R in‐place
        U_gpu[i_arr, i_arr + 1] = t.astype(U_gpu.dtype)
        R_gpu[i_arr, i_arr + 1] += u * t

    # # 2) Main reduction loops
    Uf_gpu = U_gpu.astype(R_gpu.dtype, copy=True)  # float‐work copy

    m = len(ranges)
    level = 0
    offset = 0

    for batch_size, current_ranges, max_h, max_w  in batches:
        if batch_size > 1 and not (batch_size < min_batch and max_h*max_w < area_batch):
            ranges_np = ranges_np_full[offset : offset + batch_size]
            i_arr, j_arr, k_arr = ranges_np[:,0], ranges_np[:,1], ranges_np[:,2]

            # 2) préparer caches
            rows_h = _rows_h_cache.setdefault(max_h, cp.arange(max_h, dtype=cp.int32))
            cols_w = _cols_w_cache.setdefault(max_w, cp.arange(max_w, dtype=cp.int32))

            # 3) construire I, J
            I = i_arr[:, None] + rows_h[None, :]   # (batch_size, max_h)
            J = j_arr[:, None] + cols_w[None, :]   # (batch_size, max_w)

            # 4) gather sous-blocs
            R11 = R_gpu[I[:,:,None], I[:,None,:]]      # (b, h, h)
            R12 = R_gpu[I[:,:,None], J[:,None,:]]      # (b, h, w)
            U22 = Uf_gpu[J[:,:,None], J[:,None,:]]     # (b, w, w)

            b, h, w = int(batch_size), int(max_h), int(max_w)
            S, U, Z = _get_workspace(dtype, b, h, w)
            # 5) GEMM + solve -> U12 (b, h, w)
            cp.matmul(R12, U22, out=S)
            cp.rint(-cp.linalg.solve(R11, S), out=U) # solve_triangular in pre-realase cupy

            # 6) mise à jour des sous-blocs
            Uf_block    = Uf_gpu[I[:,:,None], I[:,None,:]]  # (b, h, h)
            cp.matmul(R11, U, out=Z)               # (b, h, w)
            cp.matmul(Uf_block, U, out=U)          # (b, h, w)
            R_gpu[I[:, :, None], J[:, None, :]]  = S + Z
            Uf_gpu[I[:, :, None], J[:, None, :]] = U
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
                Uf_gpu[i:j, j:k] = Uf_gpu[i:j, i:j].dot(U12p)
        offset+= batch_size
    U_gpu[:, :] = cp.rint(Uf_gpu).astype(U_gpu.dtype)