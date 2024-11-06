#ifndef ENUMLIB_WRAPPER_ENUMERATION_CPP
#define ENUMLIB_WRAPPER_ENUMERATION_CPP

#include "enumeration.hpp"

/*
 * Perform enumeration to solve SVP in dimension N, using the enumlib library by Marc Stevens.
 *
 * @param N is dimension
 * @param R: upper-diagonal matrix of dimension N*N. B=Q*R
 * @param rowstride: rowstride of R. R(row,col) = R[rowstride*row + col]
 * @param pruningvector: vector of dimension N containing bounds for the squared norm within the projected sublattices.
 * @param sol: return param: integer vector solution with respect to current basis, or the 0 vector otherwise
 *
 * Complexity: exponential in N.
 */
FT enumeration(const int N, const FT *R, const int rowstride, const FT* pruningvector, ZZ* sol)
{
    // ensure we always return the 0-vector in sol, unless a valid solution is found
    std::fill(sol, sol+N, ZZ(0));

    // enumeration is only supported up to MAX_ENUM_N dimensions
    if (N > MAX_ENUM_N || N <= 0)
        return 0.0;

    // we pad the enumeration tree with virtual basis vectors up to dim MAX_ENUM_N
    // these virtual basis vectors have very high length
    // thus these will have zero coefficients in any found solution
    lattice_enum_t<MAX_ENUM_N> enumobj;

    // initialize enumobj.muT
    // assumption: enumobj.muT is all-zero
    for (int i = 0; i < N-1; ++i)
    {
        FT* muTi = &enumobj.muT[i][0];
        const FT* Ri = R+(i*rowstride);
        FT Rii_inv = FT(1.0) / Ri[i];
        for (int j = i+1; j < N; ++j)
        {
            // muT[i][j] = <bj,bi*> / ||bi*||^2
            muTi[j] = Ri[j] * Rii_inv;
        }
    }

    // initialize enumobj.risq and enumobj.pr
    for (int i = 0; i < N; ++i)
    {
        // risq[i] = ||bi*||^2
        enumobj.risq[i] = R[i*rowstride+i] * R[i*rowstride+i];
        // ensure 0 <= pr[i] <= ||b0*||^2
        enumobj.pr[i] = std::min<FT>( enumobj.risq[0], std::max<FT>(0.0, pruningvector[i]) );
    }

    // pad enumeration tree to MAX_ENUM_N dimension using virtual basis vectors of length above enumeration bound
    for (int i = N; i < MAX_ENUM_N; ++i)
    {
        // ensure these virtual basis vectors are never used
        enumobj.risq[i] = 2.0 * enumobj.risq[0]; // = 2 * ||b0*||^2
        enumobj.pr[i] = enumobj.pr[N-1]; // <= ||b0*||^2
    }

    // perform enumeration
    enumobj.enumerate_recursive();

    // the virtual basis vectors should never be used
    // if sol is non-zero for these positions then there is an internal error
    for (int i = N; i < MAX_ENUM_N; ++i)
        if (enumobj._sol[i] != 0)
        {
            std::cerr << "[enum]: dim=" << N << ": internal error _sol[" << i << "] != 0." << std::endl;
            return 0.0;
        }

    // write enumeration solution to sol
    for (int i = 0; i < N; ++i)
	{
        sol[i] = enumobj._sol[i];
	}

	// return the squared norm of the solution found
	const FT frac = 1.0 - (1.0/1024.0);
	return (enumobj.pr[0] < enumobj.risq[0] * frac) ? enumobj.pr[0] : 0.0;
}

#endif // ENUMLIB_WRAPPER_ENUMERATION_CPP
