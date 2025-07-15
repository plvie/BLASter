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

def seysen_reduce_gpu(R_gpu, U_gpu):
    """
    GPU version of Seysen's reduction on an upper-triangular matrix R_gpu,
    tracking the transformation in U_gpu. Both inputs are modified in place.

    :param R_gpu: cp.ndarray, upper-triangular matrix to reduce (float dtype)
    :param U_gpu: cp.ndarray, integer transformation matrix (upper-triangular with 1s on diag)
    """
    n = R_gpu.shape[0]

    # Compute your base_cases and ranges on the host (they're small)
    base_cases, ranges = __reduction_ranges(n)

    # 1) Handle the simple adjacent swaps
    # for i in base_cases:
    #     # t = -round(R[i,i+1] / R[i,i])
    #     t = -cp.rint(R_gpu[i, i+1] / R_gpu[i, i])
    #     U_gpu[i, i+1] = t.astype(U_gpu.dtype)
    #     R_gpu[i, i+1] += R_gpu[i, i] * t

    if base_cases:
        i_arr = cp.array(base_cases, dtype=cp.int32)
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
    #param ici
    min_batch = 32
    level = 1
    offset = 0
    while True:
        # calculer la taille du batch courant
        max_w = 2**level
        # max_w = max((k - i) for (i, j, k) in current_ranges) # si pas dim = 2**n enft il faudrait batch size, until grow size each time
        batch_size = m // max_w + 1

        if batch_size <= min_batch:
            break

        # on prend les batch_size prochains ranges
        current_ranges = ranges[offset : offset + batch_size]

        # déterminer la largeur max pour le padding
        # allouer les cubes 3D
        R11 = cp.zeros((batch_size, max_w, max_w), dtype=R_gpu.dtype)
        R12 = cp.zeros((batch_size, max_w, max_w), dtype=R_gpu.dtype)
        U22 = cp.zeros((batch_size, max_w, max_w), dtype=R_gpu.dtype)

        # remplir
        for idx, (i, j, k) in enumerate(current_ranges):
            w1, w2 = j - i, k - j
            R11[idx, :w1, :w1] = R_gpu[i:j, i:j]
            R12[idx, :w1, :w2] = R_gpu[i:j, j:k]
            U22[idx, :w2, :w2] = Uf_gpu[j:k, j:k]

        # batched GEMM + solve
        S12 = cp.matmul(R12, U22)
        U12 = cp.rint(-cp.linalg.solve(R11, S12))



                # mise à jour
        U12_batch  = U12[:, :max_w, :max_w]     # (N, max_w, max_w)
        R11_batch  = R11[:, :max_w, :max_w]     # (N, max_w, max_w)

        N = len(current_ranges)
        Uf_block_batch = cp.empty((N, max_w, max_w), dtype=Uf_gpu.dtype)

        # Remplir les blocs (gather manuelle)
        for idx, (i, j, k) in enumerate(current_ranges):
            Uf_block_batch[idx] = Uf_gpu[i:j, i:j]

        # 2. Matmul batched
        R12_batch  = cp.matmul(R11_batch, U12_batch)   # (N, max_w, max_w)

        # 3. Même chose pour Uf_gpu[i:j, j:k]
        Uf12_batch = cp.matmul(Uf_block_batch[:, :max_w, :max_w], U12_batch)

        # 4. Copy dans les slices finales
        for idx, (i, j, k) in enumerate(current_ranges):
            R_gpu[i:j, j:k] = S12[idx] + R12_batch[idx]
            Uf_gpu[i:j, j:k] = Uf12_batch[idx]


        # # mise à jour
        # print("test")
        # for idx, (i, j, k) in enumerate(current_ranges):
        #     w1, w2 = j - i, k - j
        #     print("w1,w2", w1, w2)
        #     s12 = S12[idx, :w1, :w2]
        #     u12 = U12[idx, :w1, :w2]
        #     R_gpu[i:j, j:k]   = s12 + R11[idx, :w1, :w1].dot(u12)
        #     Uf_gpu[i:j, j:k] = Uf_gpu[i:j, i:j].dot(u12)

        # avancer l'offset et augmenter le niveau
        offset += batch_size
        level += 1
    for (i, j, k) in ranges[offset:]:
        # S12' = R[i:j, j:k] @ U[j:k, j:k]
        S12p = R_gpu[i:j, j:k].dot(Uf_gpu[j:k, j:k])

        # U12' = round(-solve(R[i:j, i:j], S12'))
        S11 = R_gpu[i:j, i:j]
        U12p = cp.rint(-solve_triangular(S11, S12p))
        Uf_gpu[i:j, j:k] = U12p

        # R[i:j, j:k] = S12' + S11 @ U12'
        R_gpu[i:j, j:k] = S12p + S11.dot(U12p)

        # U[i:j, j:k] = U[i:j, i:j] @ U12'
        Uf_gpu[i:j, j:k] = Uf_gpu[i:j, i:j].dot(U12p)
    U_gpu[:, :] = cp.rint(Uf_gpu).astype(U_gpu.dtype)