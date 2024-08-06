# distutils: language = c++

# integer type
ctypedef long long ZZ

cdef extern from "eigen_matmul.cpp":
    void eigen_matmul(const ZZ *a, const ZZ *b, ZZ *c, int n, int m, int k) noexcept nogil
    void eigen_right_matmul(ZZ *a, const ZZ *b, int n, int m) noexcept nogil
    void eigen_right_matmul_strided(ZZ *a, const ZZ *b, int n, int m, int stride_a, int stride_b) noexcept nogil
    void eigen_init(int num_cores) noexcept nogil
