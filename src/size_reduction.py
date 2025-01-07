"""
Utility functions for calling Babai's nearest plane algorithm, size-reducing a basis or
Seysen-reducing a basis.

In comments, the old recursive functions are kept for clarity.
"""
import numpy as np

# Local imports
from seysen_lll import ZZ_matmul_strided, ZZ_left_matmul_strided, FT_matmul


# Reduction properties:


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
    Return whether R is LLL-reduced (weakly-LLL & size-reduced)
    :param R: upper-triangular matrix
    :param delta: delta-factor used in the Lovasz condition
    :return: bool
    """
    return is_weakly_lll_reduced(R, delta) and is_size_reduced(R)


# Reduction algorithms

def nearest_plane(R, T, U):
    """
    Perform Babai's Nearest Plane algorithm on multiple targets (all the columns of T), with
    respect to the upper-triangular basis R.
    This function updates T <- T + RU such that `T + RU` is in the fundamental Babai domain.
    Namely, |(T + RU)_{ij}| <= 0.5 R_ii.

    Complexity: O(N n^{omega-1}) if R is a `n x n` matrix, T is a `n x N` matrix, and `N >= n`.

    :param R: upper-triangular basis of a lattice.
    :param T: matrix containing many targets requiring reduction.
    :param U: the output transformation used to reduce T wrt R.
    :return: Nothing! The result is in T and U.
    """
    n = len(R)
    for j in range(n-1, -1, -1):
        U[j, :] = -np.rint((1.0 / R[j, j]) * T[j, :]).astype(np.int64)
        T[j, :] += R[j, j] * U[j, :]

        # apply reduction of [i:i+w) on the coefficients [i-w:i).
        if j & 1:
            T[j-1, :] += R[j-1, j] * U[j, :].astype(np.float64)
        else:
            w = (j & -j)
            i, k = j - w, min(n, j + w)
            # R11, R12 = R[i:j, i:j], R[i:j, j:k]
            # T1, T2 = T[i:j, :], T[j:k, :]

            # T1 = T1 + R12 · U2
            T[i:j, :] += FT_matmul(R[i:j, j:k], U[j:k, :].astype(np.float64))


def size_reduce(R, U):
    """
    Perform size reduction on R *inplace*, and write the transformation done to R in U, such that
    calling this function with (R, U) will update the value R to R' = RU.

    Complexity: O(n^omega) for a `n x n` matrix R.

    :param R: upper-triangular basis of a lattice.
    :param U: the matrix U to store the transformation *applied* to R.
              U will be upper triangular with unit diagonal.
    :return: Nothing! R is size reduced in place.
    """
    # Assume diag(U) = (1, 1, ..., 1).
    n = len(R)
    for i in range(0, n-1, 2):
        U[i, i + 1] = -round(R[i, i + 1] / R[i, i])
        R[i, i + 1] += R[i, i] * U[i, i + 1]

    width, hwidth = 4, 2
    while hwidth < n:
        for i in range(0, n - hwidth, width):
            j, k = i + hwidth, min(n, i + width)
            # Size reduce [j, k) with respect to [i, j).
            #
            #     [R11 R12]      [U11 U12]              [S11 S12]
            # R = [ 0  R22], U = [ 0  U22], S = R · U = [ 0  S22]
            #
            # The previous iteration computed U11 and U22.
            # Currently, R11 and R22 contain the values of
            # S11 = R11 · U11 and S22 = R22 · U22 respectively.

            # W = R12 · U22
            R[i:j, j:k] = FT_matmul(R[i:j, j:k], U[j:k, j:k].astype(np.float64))

            # U12', S12 = NearestPlane(S11, W)
            nearest_plane(R[i:j, i:j], R[i:j, j:k], U[i:j, j:k])

            # U12 = U11 · U12'
            ZZ_left_matmul_strided(U[i:j, i:j], U[i:j, j:k])

        width, hwidth = 2 * width, width


def seysen_reduce(R, U):
    """
    Perform Seysen reduction on a matrix R, while keeping track of the transformation matrix U.
    The matrix R is updated along the way.

    :param R: an upper-triangular matrix that will be modified
    :param U: an upper-triangular transformation matrix such that diag(U) = (1, 1, ..., 1).
    :return: Nothing! R is Seysen reduced in place.
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
#        S11 = FT_matmul(R[:m, :m], U[:m, :m].astype(np.float64))
#        S12 = FT_matmul(R[:m, m:], U[m:, m:].astype(np.float64))
#
#        # W = round(S11^{-1} S12).
#        W = np.rint(FT_matmul(np.linalg.inv(S11), S12)).astype(np.int64)
#        # Now take the fractional part of the entries of W.
#        U[:m, m:] = ZZ_matmul_strided(-U[:m, :m], W)
