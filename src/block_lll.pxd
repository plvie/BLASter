# distutils: language = c++
from libcpp cimport bool

# floating-point type
ctypedef double FT
# integer type
ctypedef long long ZZ

# First, these are the functions that are used to do LLL on small blocks, using
# pure C++ code:
cdef extern from "block_lll.cpp":
    bool lll_reduce(const int N, FT *R, ZZ *U, const FT delta, const size_t row_stride) noexcept nogil
