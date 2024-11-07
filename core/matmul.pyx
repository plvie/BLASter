# distutils: language = c++

import numpy as np
cimport numpy as cnp

from openmp cimport omp_set_num_threads

from decl cimport FT, ZZ
from decl cimport eigen_init, eigen_matmul, eigen_right_matmul

cnp.import_array()  # http://docs.cython.org/en/latest/src/tutorial/numpy.html#adding-types
NP_FT = np.float64  # floating-point type
NP_ZZ = np.int64  # integer type


def set_num_cores(int num_cores) -> None:
    omp_set_num_threads(num_cores)  # used by `prange` in block_X
    eigen_init(num_cores)


# ZZ (integer type)
def ZZ_matmul(const ZZ[:, ::1] A, const ZZ[:, ::1] B) -> cnp.ndarray[ZZ]:
    cdef int n = A.shape[0], m = A.shape[1], k = B.shape[1]
    cdef ZZ[:, ::1] C = np.empty(shape=(n, k), dtype=NP_ZZ)

    assert B.shape[0] == m, "Dimension mismatch"
    eigen_matmul[ZZ](<const ZZ*>&A[0, 0], <const ZZ*>&B[0, 0], &C[0, 0], n, m, k)
    return np.asarray(C)


# Variant with row stride for A:
def ZZ_matmul_strided(const ZZ[:, :] A, const ZZ[:, ::1] B) -> cnp.ndarray[ZZ]:
    cdef int n = A.shape[0], m = A.shape[1], k = B.shape[1]
    cdef int stride_a = A.strides[0] // sizeof(ZZ)

    assert B.shape[0] == m, "Dimension mismatch"
    assert A.strides[1] == sizeof(FT), "Array A is not C-contiguous"

    cdef ZZ[:, ::1] C = np.empty(shape=(n, k), dtype=NP_ZZ)

    eigen_matmul[ZZ](<const ZZ*>&A[0, 0], <const ZZ*>&B[0, 0], &C[0, 0], n, m, k, stride_a)
    return np.asarray(C)


def ZZ_right_matmul(ZZ[:, ::1] A, const ZZ[:, ::1] B) -> None:
    cdef int n = A.shape[0], m = A.shape[1]

    assert B.shape[0] == m and B.shape[1] == m, "Dimension mismatch"

    eigen_right_matmul[ZZ](<ZZ*>&A[0, 0], <const ZZ*>&B[0, 0], n, m)


# Variant with row stride for A:
# Note: B is a 1D-array of length m^2. This is used in lattice_reduction.pyx
def ZZ_right_matmul_strided(ZZ[:, :] A, const ZZ[:] B) -> None:
    cdef int n = A.shape[0], m = A.shape[1]
    cdef int stride_a = A.strides[0] // sizeof(ZZ)

    assert A.strides[1] == sizeof(ZZ), "Array A not C-contiguous"

    eigen_right_matmul[ZZ](<ZZ*>&A[0, 0], <const ZZ*>&B[0], n, m, stride_a)


# FT (floating-point type)
def FT_matmul(cnp.ndarray[FT, ndim=2] A, cnp.ndarray[FT, ndim=2] B) -> cnp.ndarray[FT]:
    # Note: NumPy uses BLAS to multiply floating-point matrices, but Eigen uses OpenMP
    return A @ B
