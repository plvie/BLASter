#include "enumeration.hpp"

/*
 * Perform enumeration
 * @param N is dimension
 * @param R: upper-diagonal matrix of dimension N*N. B=Q*R
 * @param rowstride: rowstride of R. R(row,col) = R[rowstride*row + col]
 * @param pruningvector: vector of dimension N containing bounds for the squared norm within the projected sublattices.
 * @param sol: return param: integer vector solution with respect to current basis, or the 0 vector otherwise
 *
 * Complexity: exponential
 */
#define maxN 256
void enumeration(const int N, const float_type *R, const size_t rowstride, const float_type* pruningvector, int_type* sol)
{
    if (N > maxN || N <= 0) return;

    // we pad the enumeration tree with virtual basis vectors up to dim maxN
    // these virtual basis vectors have very high length
    // thus these will have zero coefficients in any found solution
    lattice_enum_t<maxN> enumobj;
    
    size_t NN = size_t(N);
    for (size_t i = 0; i < NN-1; ++i)
    {
        for (size_t j = i+1; j < NN; ++j)
        {
            // muT[i][j] = <bi,bi*> / ||bi*||^2
            enumobj.muT[i][j] = R[i*rowstride + j] / R[i*rowstride + i];
        }
        for (size_t j = NN; j < maxN; ++j)
        {
            enumobj.muT[i][j] = 0.0;
        }
    }
    for (size_t i = NN; i < maxN; ++i)
    {
        for (size_t j = i+1; j < maxN; ++j)
        {
            enumobj.muT[i][j] = 0.0;
        }
    }
    for (size_t i = 0; i < NN; ++i)
    {
        // risq[i] = ||bi*||^2
        enumobj.risq[i] = R[i*rowstride+i] * R[i*rowstride+i];
        // ensure 0 <= pr[i] <= ||b0*||^2
        enumobj.pr[i] = std::min<float_type>( enumobj.risq[0], std::max<float_type>(0.0, pruningvector[i]) );
    }
    for (size_t i = NN; i < maxN; ++i)
    {
        // ensure these virtual basis vectors are never used
        enumobj.risq[i] = 2.0 * enumobj.risq[0];
        enumobj.pr[i] = enumobj.pr[NN-1];
    }
    enumobj.enumerate_recursive<true>();
    for (size_t i = 0; i < NN; ++i)
        sol[i] = enumobj._sol[i];
}
