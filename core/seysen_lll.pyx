# distutils: language = c++

import numpy as np

cimport cython
cimport numpy as cnp

from cysignals.signals cimport sig_on, sig_off
from cython.parallel cimport prange
from libc.string cimport memcpy
from openmp cimport omp_get_num_threads, omp_get_thread_num, omp_set_num_threads

from decl cimport FT, ZZ, lll_reduce, deeplll_reduce, bkz_reduce
from decl cimport eigen_init, eigen_matmul, eigen_right_matmul, eigen_right_matmul_strided


# Taken from:
# http://docs.cython.org/en/latest/src/tutorial/numpy.html#adding-types
cnp.import_array()
NP_FT = np.float64  # floating-point type
NP_ZZ = np.int64  # integer type


@cython.boundscheck(False)
@cython.wraparound(False)
def block_lll(
        cnp.ndarray[FT, ndim=2] R, cnp.ndarray[ZZ, ndim=2] B_red, cnp.ndarray[ZZ, ndim=2] U,
        FT delta, int offset, int block_size) -> None:

    # Variables
    cdef Py_ssize_t n = R.shape[0]
    cdef int i, j, w, num_blocks = int((n - offset + block_size - 1) / block_size), block_id
    cdef FT[:, ::1] R_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_FT)
    cdef ZZ[:, ::1] U_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_ZZ)

    # Check that these are of the correct type:
    assert R.dtype == NP_FT and U.dtype == NP_ZZ

    sig_on()
    for block_id in prange(num_blocks, nogil=True):
        i = offset + block_size * block_id
        w = min(n - i, block_size)

        for j in range(w):
            memcpy(&R_sub[block_id, j * w], &R[i + j, i], w * sizeof(FT));

        # Step 1: run LLL on block [i, i + w).
        lll_reduce(w, &R_sub[block_id, 0], &U_sub[block_id, 0], delta)

        for j in range(w):
            memcpy(&R[i + j, i], &R_sub[block_id, j * w], w * sizeof(FT));

    sig_off()
    # Step 2: Update U and B_red locally by multiplying with U_sub[block_id].
    for block_id in range(num_blocks):
        i = offset + block_size * block_id
        w = min(n - i, block_size)
        eigen_right_matmul_strided(<ZZ*>&U[0, i], <const ZZ*>&U_sub[block_id, 0], n, w, n)
        eigen_right_matmul_strided(<ZZ*>&B_red[0, i], <const ZZ*>&U_sub[block_id, 0], n, w, n)


@cython.boundscheck(False)
@cython.wraparound(False)
def block_deep_lll(
        cnp.ndarray[FT, ndim=2] R, cnp.ndarray[ZZ, ndim=2] B_red, cnp.ndarray[ZZ, ndim=2] U,
        FT delta, int offset, int block_size, int depth) -> None:

    # Variables
    cdef Py_ssize_t n = R.shape[0]
    cdef int i, j, w, num_blocks = int((n - offset + block_size - 1) / block_size), block_id
    cdef FT[:, ::1] R_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_FT)
    cdef ZZ[:, ::1] U_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_ZZ)

    # Check that these are of the correct type:
    assert R.dtype == NP_FT and U.dtype == NP_ZZ

    sig_on()
    for block_id in prange(num_blocks, nogil=True):
        i = offset + block_size * block_id
        w = min(n - i, block_size)

        for j in range(w):
            memcpy(&R_sub[block_id, j * w], &R[i + j, i], w * sizeof(FT));

        # Step 1: run DeepLLL on block [i, i + w).
        deeplll_reduce(w, &R_sub[block_id, 0], &U_sub[block_id, 0], delta, depth)

        for j in range(w):
            memcpy(&R[i + j, i], &R_sub[block_id, j * w], w * sizeof(FT));

    sig_off()
    # Step 2: Update U and B_red locally by multiplying with U_sub[block_id].
    for block_id in range(num_blocks):
        i = offset + block_size * block_id
        w = min(n - i, block_size)
        eigen_right_matmul_strided(<ZZ*>&U[0, i], <const ZZ*>&U_sub[block_id, 0], n, w, n)
        eigen_right_matmul_strided(<ZZ*>&B_red[0, i], <const ZZ*>&U_sub[block_id, 0], n, w, n)


@cython.boundscheck(False)
@cython.wraparound(False)
def block_bkz(
        cnp.ndarray[FT, ndim=2] R, cnp.ndarray[ZZ, ndim=2] B_red, cnp.ndarray[ZZ, ndim=2] U,
        FT delta, int offset, int block_size, int beta, int max_tours = 0) -> None:

    # Variables
    cdef Py_ssize_t n = R.shape[0]
    cdef int i, j, w, num_blocks = int((n - offset + block_size - 1) / block_size), block_id
    cdef FT[:, ::1] R_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_FT)
    cdef ZZ[:, ::1] U_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_ZZ)

    # Check that these are of the correct type:
    assert R.dtype == NP_FT and U.dtype == NP_ZZ

    sig_on()
    for block_id in prange(num_blocks, nogil=True):
        i = offset + block_size * block_id
        w = min(n - i, block_size)

        for j in range(w):
            memcpy(&R_sub[block_id, j * w], &R[i + j, i], w * sizeof(FT));

        # Step 1: run BKZ on block [i, i + w).
        bkz_reduce(w, &R_sub[block_id, 0], &U_sub[block_id, 0], delta, beta, max_tours)

        for j in range(w):
            memcpy(&R[i + j, i], &R_sub[block_id, j * w], w * sizeof(FT));

    sig_off()
    # Step 2: Update U and B_red locally by multiplying with U_sub[block_id].
    for block_id in range(num_blocks):
        i = offset + block_size * block_id
        w = min(n - i, block_size)
        eigen_right_matmul_strided(<ZZ*>&U[0, i], <const ZZ*>&U_sub[block_id, 0], n, w, n)
        eigen_right_matmul_strided(<ZZ*>&B_red[0, i], <const ZZ*>&U_sub[block_id, 0], n, w, n)


def set_num_cores(int num_cores) -> None:
    omp_set_num_threads(num_cores)  # used by `prange` in block_X
    eigen_init(num_cores)


@cython.boundscheck(False)
@cython.wraparound(False)
def ZZ_matmul(
        cnp.ndarray[ZZ, ndim=2, mode='c'] A,
        cnp.ndarray[ZZ, ndim=2, mode='c'] B) -> cnp.ndarray[ZZ]:
    cdef int n = A.shape[0], m = A.shape[1], k = B.shape[1]
    assert B.shape[0] == m, "Dimension mismatch"
    cdef ZZ[:, ::1] C = np.empty(shape=(n, k), dtype=NP_ZZ)

    eigen_matmul(<const ZZ*>&A[0, 0], <const ZZ*>&B[0, 0], &C[0, 0], n, m, k)
    return np.asarray(C)


@cython.boundscheck(False)
@cython.wraparound(False)
def ZZ_right_matmul(
        cnp.ndarray[ZZ, ndim=2, mode='c'] A,
        cnp.ndarray[ZZ, ndim=2, mode='c'] B) -> None:
    cdef int n = A.shape[0], m = A.shape[1]
    assert B.shape[0] == m and B.shape[1] == m, "Dimension mismatch"

    eigen_right_matmul(<ZZ*>&A[0, 0], <const ZZ*>&B[0, 0], n, m)
