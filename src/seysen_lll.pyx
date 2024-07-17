# distutils: language = c++

# Taken from:
# http://docs.cython.org/en/latest/src/tutorial/numpy.html#adding-types


import numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# distributed with Numpy).
# Here we've used the name "cnp" to make it easier to understand what
# comes from the cimported module and what comes from the imported module,
# however you can use the same name for both if you wish.
cimport numpy as cnp
cimport cython
from cython.parallel cimport prange

from block_lll cimport FT, ZZ, lll_reduce
from eigen_matmul cimport eigen_init as _eigen_init, eigen_matmul as _eigen_matmul, eigen_right_matmul as _eigen_right_matmul

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. Therefore we recommend
# always calling "import_array" whenever you "cimport numpy"
cnp.import_array()

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE_ZZ = np.int64
DTYPE_FT = np.float64

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef cnp.int64_t DTYPE_ZZ_t
ctypedef cnp.float64_t DTYPE_FT_t


@cython.boundscheck(False)
@cython.wraparound(False)
def perform_lll_on_blocks(
        cnp.ndarray[DTYPE_FT_t, ndim=2] R,
        cnp.ndarray[DTYPE_ZZ_t, ndim=2] U,
        FT delta,
        int offset,
        int block_size) -> bool:

    cdef Py_ssize_t n = R.shape[0]
    cdef int idx, i, j, k, w
    cdef n_blocks = (n - offset + block_size - 1) // block_size
    cdef FT[:, ::1] R_sub = np.empty(shape=(block_size, n), dtype=DTYPE_FT)
    cdef ZZ[:, ::1] U_sub = np.zeros(shape=(block_size, n), dtype=DTYPE_ZZ)

    # Check that these are of the correct type:
    assert R.dtype == DTYPE_FT and U.dtype == DTYPE_ZZ

    for i in range(offset, n, block_size):
        w = min(n - i, block_size)
        for j in range(w):
            for k in range(i, i + w):
                R_sub[j, k] = R[i + j, k]

    cdef int res = 0
    for i in prange(offset, n, block_size, nogil=True):
        j = min(n, i + block_size)
        if not lll_reduce(j - i, &R_sub[0, i], &U_sub[0, i], delta, n):
            res = 1
    if res == 1:
        return False

    # TODO: Make the part of U that needs transforming continuous,
    # such that the matrix multiplication can be done in parallel:
    for i in range(offset, n, block_size):
        j = min(n, i + block_size)
        U[:, i:j] = U[:, i:j] @ np.asarray(U_sub)[0:j - i, i:j]
    return True


@cython.boundscheck(False)
@cython.wraparound(False)
def eigen_matmul(cnp.ndarray[DTYPE_ZZ_t, ndim=2, mode='c'] A, cnp.ndarray[DTYPE_ZZ_t, ndim=2, mode='c'] B) -> cnp.ndarray[DTYPE_ZZ_t]:
    cdef int n = A.shape[0]
    cdef int m = A.shape[1]
    cdef int k = B.shape[1]
    assert B.shape[0] == m, "Dimension mismatch"
    cdef ZZ[:, ::1] C = np.empty(shape=(n, k), dtype=DTYPE_ZZ)

    _eigen_matmul(<const ZZ*>&A[0, 0], <const ZZ*>&B[0, 0], &C[0, 0], n, m, k)
    return np.asarray(C)


@cython.boundscheck(False)
@cython.wraparound(False)
def eigen_right_matmul(cnp.ndarray[DTYPE_ZZ_t, ndim=2, mode='c'] A, cnp.ndarray[DTYPE_ZZ_t, ndim=2, mode='c'] B) -> None:
    cdef int n = A.shape[0]
    assert A.shape[1] == n and B.shape[0] == n and B.shape[1] == n, "Dimension mismatch"

    _eigen_right_matmul(<ZZ*>&A[0, 0], <const ZZ*>&B[0, 0], n)


def eigen_init(int num_cores=0):
    _eigen_init(num_cores)
