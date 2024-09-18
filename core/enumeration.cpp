#include "enumeration.hpp"

/*
 * Perform enumeration
 * @param N is dimension
 * @param R: upper-diagonal matrix of dimension N*N. B=Q*R
 * @param rowstride: rowstride of R. R(row,col) = R[rowstride*row + col]
 * @param pruningvector: vector of dimension N containing squared norm bounds of projected sublattices.
 * @param sol: return param: integer vector solution with respect to current basis
 *
 * Complexity: exponential
 */
void enumeration(const int N, const float_type *R, const size_t rowstride, const float_type* pruningvector, int_type* sol)
{
    lattice_enum_t<16> enumobj;
    if (N != 16) return;
    
    size_t NN = size_t(N);
    for (size_t i = 0; i < NN-1; ++i)
    {
        for (size_t j = i+1; j < NN; ++j)
        {
            enumobj.muT[i][j] = R[i*rowstride + j] / R[i*rowstride + i];
        }
    }
    for (size_t i = 0; i < NN; ++i)
    {
        enumobj.risq[i] = R[i*rowstride+i]*R[i*rowstride+i];
        enumobj.pr[i] = pruningvector[i];
    }
    enumobj.enumerate_recursive<true>();
    for (size_t i = 0; i < NN; ++i)
        sol[i] = enumobj._sol[i];
}
