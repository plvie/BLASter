# distutils: language = c++

import numpy as np
cimport cython
cimport numpy as cnp

from cysignals.signals cimport sig_on, sig_off
from cython.parallel cimport prange
from libc.string cimport memcpy
from openmp cimport omp_set_num_threads, omp_get_num_threads, omp_get_thread_num

from decl cimport FT, ZZ, lll_reduce, deeplll_reduce, bkz_reduce, \
    eigen_init, eigen_matmul, eigen_left_matmul, eigen_right_matmul


cnp.import_array()  # http://docs.cython.org/en/latest/src/tutorial/numpy.html#adding-types
NP_FT = np.float64  # floating-point type
NP_ZZ = np.int64  # integer type


cdef int debug_size_reduction = 0


def set_debug_flag(int flag):
    global debug_size_reduction
    debug_size_reduction = flag


def set_num_cores(int num_cores):
    omp_set_num_threads(num_cores)  # used by `prange` in block_X
    eigen_init(num_cores)


#
# Lattice reduction
#
@cython.boundscheck(False)
@cython.wraparound(False)
def block_lll(
        cnp.ndarray[FT, ndim=2] R, cnp.ndarray[ZZ, ndim=2] B_red, cnp.ndarray[ZZ, ndim=2] U,
        FT delta, int offset, int block_size) -> None:
    global debug_size_reduction

    # Variables
    cdef:
        Py_ssize_t n = R.shape[0]
        int i, j, w, num_blocks = int((n - offset + block_size - 1) / block_size), block_id
        FT[:, ::1] R_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_FT)
        ZZ[:, ::1] U_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_ZZ)

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

        if debug_size_reduction != 0:
            for j in range(w):
                memcpy(&R[i + j, i], &R_sub[block_id, j * w], w * sizeof(FT));

    sig_off()

    # Step 2: Update U and B_red locally by multiplying with U_sub[block_id].
    for block_id in range(num_blocks):
        i = offset + block_size * block_id
        w = min(n - i, block_size)

        ZZ_right_matmul_strided(U[:, i:i+w], U_sub[block_id, 0:w*w])
        ZZ_right_matmul_strided(B_red[:, i:i+w], U_sub[block_id, 0:w*w])


@cython.boundscheck(False)
@cython.wraparound(False)
def block_deep_lll(int depth,
        cnp.ndarray[FT, ndim=2] R, cnp.ndarray[ZZ, ndim=2] B_red, cnp.ndarray[ZZ, ndim=2] U,
        FT delta, int offset, int block_size) -> None:
    global debug_size_reduction

    # Variables
    cdef:
        Py_ssize_t n = R.shape[0]
        int i, j, w, num_blocks = int((n - offset + block_size - 1) / block_size), block_id
        FT[:, ::1] R_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_FT)
        ZZ[:, ::1] U_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_ZZ)

    # Check that these are of the correct type:
    assert R.dtype == NP_FT and U.dtype == NP_ZZ

    sig_on()
    for block_id in prange(num_blocks, nogil=True):
        i = offset + block_size * block_id
        w = min(n - i, block_size)

        for j in range(w):
            memcpy(&R_sub[block_id, j * w], &R[i + j, i], w * sizeof(FT));

        # Step 1: run deep-LLL on block [i, i + w).
        deeplll_reduce(w, &R_sub[block_id, 0], &U_sub[block_id, 0], delta, depth)

        if debug_size_reduction != 0:
            for j in range(w):
                memcpy(&R[i + j, i], &R_sub[block_id, j * w], w * sizeof(FT));

    sig_off()

    # Step 2: Update U and B_red locally by multiplying with U_sub[block_id].
    for block_id in range(num_blocks):
        i = offset + block_size * block_id
        w = min(n - i, block_size)

        ZZ_right_matmul_strided(U[:, i:i+w], U_sub[block_id, 0:w*w])
        ZZ_right_matmul_strided(B_red[:, i:i+w], U_sub[block_id, 0:w*w])


@cython.boundscheck(False)
@cython.wraparound(False)
def block_bkz(int beta,
        cnp.ndarray[FT, ndim=2] R, cnp.ndarray[ZZ, ndim=2] B_red, cnp.ndarray[ZZ, ndim=2] U,
        FT delta, int offset, int block_size) -> None:
    global debug_size_reduction

    # Variables
    cdef:
        Py_ssize_t n = R.shape[0]
        int i, j, w, num_blocks = int((n - offset + block_size - 1) / block_size), block_id
        FT[:, ::1] R_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_FT)
        ZZ[:, ::1] U_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_ZZ)

    # Check that these are of the correct type:
    assert R.dtype == NP_FT and U.dtype == NP_ZZ

    sig_on()
    for block_id in prange(num_blocks, nogil=True):
        i = offset + block_size * block_id
        w = min(n - i, block_size)

        for j in range(w):
            memcpy(&R_sub[block_id, j * w], &R[i + j, i], w * sizeof(FT));

        # Step 1: run BKZ on block [i, i + w).
        bkz_reduce(w, &R_sub[block_id, 0], &U_sub[block_id, 0], delta, beta)

        if debug_size_reduction != 0:
            for j in range(w):
                memcpy(&R[i + j, i], &R_sub[block_id, j * w], w * sizeof(FT));

    sig_off()

    # Step 2: Update U and B_red locally by multiplying with U_sub[block_id].
    for block_id in range(num_blocks):
        i = offset + block_size * block_id
        w = min(n - i, block_size)

        ZZ_right_matmul_strided(U[:, i:i+w], U_sub[block_id, 0:w*w])
        ZZ_right_matmul_strided(B_red[:, i:i+w], U_sub[block_id, 0:w*w])

@cython.boundscheck(False)
@cython.wraparound(False)
def get_R_sub_HKZ(cnp.ndarray[FT, ndim=2] R, int cur_front, int w):
    cdef int n = R.shape[0]
    cdef cnp.ndarray[FT, ndim=2] R_sub = np.empty((w, w), dtype=np.float64)
    cdef FT[:, ::1] R_mv = R
    cdef FT[:, ::1] R_sub_mv = R_sub
    cdef size_t row_bytes = w * sizeof(FT)
    cdef int j
    for j in range(w):
        memcpy(&R_sub_mv[j, 0], &R_mv[cur_front + j, cur_front], row_bytes)
    return R_sub

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_U_HKZ(cnp.ndarray[ZZ, ndim=2] B_red,
                cnp.ndarray[ZZ, ndim=2] U,
                cnp.ndarray[ZZ, ndim=2] U_sub,
                int cur_front, int w):
    """
    Applique U_sub (wxw) aux colonnes [cur_front:cur_front+w) de U et B_red,
    en-place, via ZZ_right_matmul_strided.
    """
    cdef int i = cur_front

    # Vues strided sur les blocs de colonnes de U et B_red
    cdef ZZ[:, :] U_block = U[:, i : i+w]
    cdef ZZ[:, :] B_block = B_red[:, i : i+w]

    # --- étape clé : construire un buffer 1D C-contigu de U_sub ---
    # 1) Copie C-contiguë en 2D
    cdef cnp.ndarray[ZZ, ndim=2] U_sub_contig = np.ascontiguousarray(U_sub)
    # 2) Remodelage  en 1D sans changer les données
    cdef cnp.ndarray[ZZ, ndim=1] U_sub_flat_arr = U_sub_contig.reshape((w*w,))
    # 3) Memoryview 1D à partir du ndarray
    cdef ZZ[:] Usub_flat = U_sub_flat_arr

    # On a maintenant :
    #   - U_block et B_block : vues sur U et B_red, chacune avec un row-stride ≠ w
    #   - Usub_flat      : un tableau 1D C-contigu de longueur w*w

    # Produit A_block ← A_block · U_sub pour A = U_block, puis B_block
    ZZ_right_matmul_strided(U_block, Usub_flat)
    ZZ_right_matmul_strided(B_block, Usub_flat)


#
# Lattice reduction GPU
#
@cython.boundscheck(False)
@cython.wraparound(False)
def block_lll_gpu(
        cnp.ndarray[FT, ndim=2] R,
        FT delta, int offset, int block_size) -> cnp.ndarray[ZZ]:
    global debug_size_reduction

    # Variables
    cdef:
        Py_ssize_t n = R.shape[0]
        int i, j, w, num_blocks = int((n - offset + block_size - 1) / block_size), block_id
        FT[:, ::1] R_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_FT)
        ZZ[:, ::1] U_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_ZZ)

    sig_on()
    for block_id in prange(num_blocks, nogil=True):
        i = offset + block_size * block_id
        w = min(n - i, block_size)

        for j in range(w):
            memcpy(&R_sub[block_id, j * w], &R[i + j, i], w * sizeof(FT));

        # Step 1: run LLL on block [i, i + w).
        lll_reduce(w, &R_sub[block_id, 0], &U_sub[block_id, 0], delta)

        if debug_size_reduction != 0:
            for j in range(w):
                memcpy(&R[i + j, i], &R_sub[block_id, j * w], w * sizeof(FT));

    sig_off()
    return U_sub

@cython.boundscheck(False)
@cython.wraparound(False)
def block_deep_lll_gpu(int depth,
        cnp.ndarray[FT, ndim=2] R,
        FT delta, int offset, int block_size) -> None:
    global debug_size_reduction

    # Variables
    cdef:
        Py_ssize_t n = R.shape[0]
        int i, j, w, num_blocks = int((n - offset + block_size - 1) / block_size), block_id
        FT[:, ::1] R_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_FT)
        ZZ[:, ::1] U_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_ZZ)

    # Check that these are of the correct type:

    sig_on()
    for block_id in prange(num_blocks, nogil=True):
        i = offset + block_size * block_id
        w = min(n - i, block_size)

        for j in range(w):
            memcpy(&R_sub[block_id, j * w], &R[i + j, i], w * sizeof(FT));

        # Step 1: run deep-LLL on block [i, i + w).
        deeplll_reduce(w, &R_sub[block_id, 0], &U_sub[block_id, 0], delta, depth)

        if debug_size_reduction != 0:
            for j in range(w):
                memcpy(&R[i + j, i], &R_sub[block_id, j * w], w * sizeof(FT));

    sig_off()

    # Step 2: Update U and B_red locally by multiplying with U_sub[block_id].
    return U_sub

@cython.boundscheck(False)
@cython.wraparound(False)
def block_bkz_gpu(int beta,
        cnp.ndarray[FT, ndim=2] R,
        FT delta, int offset, int block_size) -> cnp.ndarray[ZZ]:
    global debug_size_reduction

    # Variables
    cdef:
        Py_ssize_t n = R.shape[0]
        int i, j, w, num_blocks = int((n - offset + block_size - 1) / block_size), block_id
        FT[:, ::1] R_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_FT)
        ZZ[:, ::1] U_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_ZZ)

    sig_on()
    for block_id in prange(num_blocks, nogil=True):
        i = offset + block_size * block_id
        w = min(n - i, block_size)

        for j in range(w):
            memcpy(&R_sub[block_id, j * w], &R[i + j, i], w * sizeof(FT));

        # Step 1: run BKZ on block [i, i + w).
        bkz_reduce(w, &R_sub[block_id, 0], &U_sub[block_id, 0], delta, beta)

        if debug_size_reduction != 0:
            for j in range(w):
                memcpy(&R[i + j, i], &R_sub[block_id, j * w], w * sizeof(FT));

    sig_off()

    return U_sub

#
# Integer (int64) Matrix Multiplication using Eigen, which internally uses OpenMP.
#
def ZZ_matmul(const ZZ[:, ::1] A, const ZZ[:, ::1] B) -> cnp.ndarray[ZZ]:
    """
    Return A * B.
    A and B should be C-contiguous.
    """
    cdef:
        int n = A.shape[0], m = A.shape[1], k = B.shape[1]
        ZZ[:, ::1] C = np.empty(shape=(n, k), dtype=NP_ZZ)

    assert B.shape[0] == m, "Dimension mismatch"
    eigen_matmul(<const ZZ*>&A[0, 0], <const ZZ*>&B[0, 0], &C[0, 0], n, m, k)
    return np.asarray(C)


def ZZ_left_matmul_strided(const ZZ[:, :] A, ZZ[:, :] B) -> None:
    """
    Compute B <- A * B.
    A and B may have a row-stride.
    """
    cdef:
        int n = B.shape[0], m = B.shape[1]
        int stride_a = A.strides[0] // sizeof(ZZ), stride_b = B.strides[0] // sizeof(ZZ)

    assert A.strides[1] == sizeof(ZZ), "Array A not C-contiguous"
    assert B.strides[1] == sizeof(ZZ), "Array B not C-contiguous"
    eigen_left_matmul(<const ZZ*>&A[0, 0], <ZZ*>&B[0, 0], n, m, stride_a, stride_b)


def ZZ_right_matmul(ZZ[:, ::1] A, const ZZ[:, ::1] B) -> None:
    """
    Compute A <- A * B.
    A and B should be C-contiguous.
    """
    cdef:
        int n = A.shape[0], m = A.shape[1]

    assert B.shape[0] == m and B.shape[1] == m, "Dimension mismatch"
    eigen_right_matmul(<ZZ*>&A[0, 0], <const ZZ*>&B[0, 0], n, m)


def ZZ_right_matmul_strided(ZZ[:, :] A, const ZZ[:] B) -> None:
    """
    Compute A <- A * B.
    A may have a row-stride. B should be a 1-dimensional array of length m^2,
    where m is the number of columns of A.
    """
    cdef:
        int n = A.shape[0], m = A.shape[1], stride_a = A.strides[0] // sizeof(ZZ)

    assert A.strides[1] == sizeof(ZZ), "Array A not C-contiguous"
    eigen_right_matmul(<ZZ*>&A[0, 0], <const ZZ*>&B[0], n, m, stride_a)


#
# Floating-point (double) Matrix Multiplication using NumPy, which internally uses BLAS.
#
def FT_matmul(cnp.ndarray[FT, ndim=2] A, cnp.ndarray[FT, ndim=2] B) -> cnp.ndarray[FT]:
    """
    Return A * B.
    """
    return A @ B

