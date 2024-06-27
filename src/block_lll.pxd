# distutils: language = c++
from libcpp cimport bool

# integer type
ctypedef long long ZZ

# First, these are the functions that are used to do LLL on small blocks, using
# pure C++ code:
cdef extern from "block_lll.cpp":
    bool lll_reduce[int](double *R, ZZ *U, double delta, size_t row_stride) noexcept nogil
    bool lll_reduce_n(int n, double *R, ZZ *U, double delta, size_t row_stride) noexcept nogil
