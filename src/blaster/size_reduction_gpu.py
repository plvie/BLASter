"""
Utility functions for calling Babai's nearest plane algorithm, size-reducing a basis or
Seysen-reducing a basis.

In comments, the old recursive functions are kept for clarity.
"""
from functools import cache
import numpy as np
import cupy as cp

# Local imports
from blaster_core import ZZ_left_matmul_strided, FT_matmul


def is_weakly_lll_reduced_gpu(R, delta=0.99):
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
    return bool(cp.all(ok).item())



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
    for i in base_cases:
        # t = -round(R[i,i+1] / R[i,i])
        t = -cp.rint(R_gpu[i, i+1] / R_gpu[i, i])
        U_gpu[i, i+1] = t.astype(U_gpu.dtype)
        R_gpu[i, i+1] += R_gpu[i, i] * t

    # 2) Main reduction loops
    for (i, j, k) in ranges:
        # S12' = R[i:j, j:k] @ U[j:k, j:k]
        S12p = R_gpu[i:j, j:k].dot(U_gpu[j:k, j:k].astype(R_gpu.dtype))

        # U12' = round(-solve(R[i:j, i:j], S12'))
        S11 = R_gpu[i:j, i:j]
        U12p = cp.rint(-cp.linalg.solve(S11, S12p)).astype(U_gpu.dtype)
        # U12p = cp.rint(-cp.linalg.inv(S11).dot(S12p)).astype(U_gpu.dtype)
        U_gpu[i:j, j:k] = U12p

        # R[i:j, j:k] = S12' + S11 @ U12'
        R_gpu[i:j, j:k] = S12p + S11.dot(U12p.astype(R_gpu.dtype))

        # U[i:j, j:k] = U[i:j, i:j] @ U12'
        U_gpu[i:j, j:k] = U_gpu[i:j, i:j].dot(U12p)



# For didactical reasons, here are the recursive versions of:
# - nearest_plane,
# - size_reduce, and
# - seysen_reduce.
#
#
# def nearest_plane(R, T, U):
#     """
#     Perform Babai's Nearest Plane algorithm on multiple targets (all the columns of T), with
#     respect to the upper-triangular basis R.
#     This function updates T <- T + RU such that `T + RU` is in the fundamental Babai domain.
#     Namely, |(T + RU)_{ij}| <= 0.5 R_ii.
#
#     Complexity: O(N n^{omega-1}) if R is a `n x n` matrix, T is a `n x N` matrix, and `N >= n`.
#
#     :param R: upper-triangular basis of a lattice.
#     :param T: matrix containing many targets requiring reduction.
#     :param U: the output transformation used to reduce T wrt R.
#     :return: Nothing! The result is in T and U.
#     """
#     n, m = R.shape[0], R.shape[0] // 2
#     if n == 1:
#         U[0, :] = -np.rint((1.0 / R[0, 0]) * T).astype(np.int64)
#         T += R[0, 0] * U
#     else:
#         # R11, R12, R22 = R[:m, :m], R[:m, m:], R[m:, m:]
#         # T1, T2 = T[:m, :], T[m:, :]
#         # U1, U2 = U[:m, :], U[m:, :]
#
#         # U2 = NP(R22, T2)
#         nearest_plane(R[m:, m:], T[m:, :], U[m:, :])
#
#         # T1 = T1 + R12 · U2
#         T[:m, :] += FT_matmul(R[:m, m:], U[m:, :].astype(np.float64))
#
#         # U1 = NP(R11, T1)
#         nearest_plane(R[:m, :m], T[:m, :], U[:m, :])
#
#
# def size_reduce(R, U):
#     """
#     Perform size reduction on R *inplace*, and write the transformation done to R in U, such that
#     calling this function with (R, U) will update the value R to R' = RU.
#
#     Complexity: O(n^omega) for a `n x n` matrix R.
#
#     :param R: upper-triangular basis of a lattice.
#     :param U: the matrix U to store the transformation *applied* to R.
#               U will be upper triangular with unit diagonal.
#     :return: Nothing! R is size reduced in place.
#     """
#     n, m = R.shape[0], R.shape[0] // 2
#     if n == 1:
#         return
#
#     if n == 2:
#         U[0, 1] = -round(R[0, 1] / R[0, 0])
#         R[0, 1] += R[0, 0] * U[0, 1]
#     else:
#         # R11, R12, R22 = R[:m, :m], R[:m, m:], R[m:, m:]
#         # U11, U12, U22 = U[:m, :m], U[:m, m:], U[m:, m:]
#
#         # U11 = SizeReduce(R11)
#         size_reduce(R[:m, :m], U[:m, :m])
#
#         # U22 = SizeReduce(R22)
#         size_reduce(R[m:, m:], U[m:, m:])
#
#         # R12 = R12 · U22
#         R[:m, m:] = FT_matmul(R[:m, m:], U[m:, m:].astype(np.float64))
#
#         # U12' = NearestPlane(basis=R11', target=R12), R12 = R12 + R11' U12'
#         nearest_plane(R[:m, :m], R[:m, m:], U[:m, m:])
#
#         # Note: NP was called with the size-reduced R11' = R11 · U11.
#         # U12 = U11 · U12'
#         # U[:m, m:] = U[:m, :m] @ U[:m, m:]
#         ZZ_left_matmul_strided(U[:m, :m], U[:m, m:])
#
#
# def seysen_reduce(R, U):
#    """
#    Seysen reduce a matrix R, recursive style, and store the result in U.
#    See: Algorithm 7 from [KEF21].
#    [KEF21] P. Kircher, T. Espitau, P.-A. Fouque. Towards faster polynomial-time lattice reduction.
#    :param R: an upper-triangular matrix (having row vectors).
#    :param U: a unimodular transformation U such that RU is Seysen-Reduced.
#    :return: None! The result is stored in U.
#    """
#    n, m = len(R), len(R) // 2
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
#        # S11 = R11 · U11
#        S11 = FT_matmul(R[:m, :m], U[:m, :m].astype(np.float64))
#
#        # S12' = R12 · U22
#        S12 = FT_matmul(R[:m, m:], U[m:, m:].astype(np.float64))
#
#        # U12' = round(-S11^{-1} S12').
#        U[i:j, j:k] = np.rint(FT_matmul(-np.linalg.inv(S11), S12)).astype(np.int64)
#
#        # U12 = U11 · U12'
#        ZZ_left_matmul_strided(U[:m, :m], U[:m, m:])
