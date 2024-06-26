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

from block_lll cimport ZZ, FT, lll_reduce_n

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


# TODO: add these two lines, when a first version works!
# @cython.boundscheck(False)
# @cython.wraparound(False)
def perform_lll_on_blocks(
        cnp.ndarray[DTYPE_FT_t, ndim=2] R,
        cnp.ndarray[DTYPE_ZZ_t, ndim=2] U,
        FT delta,
        int offset,
        int block_size):

    cdef int n = R.shape[0], i, j, w
    cdef FT[:, :] R_sub
    cdef ZZ[:, :] U_sub

    # Check that these are of the correct type:
    assert R.dtype == DTYPE_FT and U.dtype == DTYPE_ZZ

    for i in range(offset, n, block_size):
        j = min(n, i + block_size)
        w = j - i

        # TODO: Alternatively, we could pass the 'stride' of `n` to the C++ level.
        R_sub = np.ascontiguousarray(R[i:j, i:j])
        U_sub = np.zeros([w, w], dtype=DTYPE_ZZ)
        if not lll_reduce_n(w, &R_sub[0, 0], &U_sub[0, 0], delta):
            return False
        U[:, i:j] = U[:, i:j] @ U_sub
    return True
