/*
MIT License

Copyright (c) 2024 Marc Stevens

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef ENUMLIB_ENUMERATION_HPP
#define ENUMLIB_ENUMERATION_HPP

#include <cmath>
#include <cstdint>
#include <array>
#include <iostream>

// floating-point type
typedef double float_type;

// integer type
typedef long long int_type;

#define NOCOUNTS 1

template <int N, bool findsubsols = false>
struct lattice_enum_t
{
	typedef std::array<float_type, N> fltrow_t;
	typedef std::array<int_type, N>   introw_t;

	/* inputs */
	// mu^T corresponds to R (B=Q*R) with multiplicative corrections: muT[i][j] = R[i][j] / R[i][i]
	// mu^T is the transposed mu in fplll (see also: An LLL Algorithm with Quadratic Complexity, Nguyen, Stehle, 2009.)
	float_type muT[N][N];
	// risq[i] is ||bi*||^2, or R[i][i]*R[i][i]
	fltrow_t risq;
	// the pruning bounds on the norms of the respective projected sublattices
	fltrow_t pr;

	/* internals */
	introw_t _x, _Dx, _D2x;
	fltrow_t _sol; // to pass to fplll
	fltrow_t _c;
	introw_t _r;
	std::array<float_type, N + 1> _l;
	std::array<std::uint64_t, N + 1> _counts;

	float_type _sigT[N][N];

	fltrow_t _subsolL;
	std::array<fltrow_t, N> _subsol;


	lattice_enum_t()
	{
	}

	inline int_type myround(double a)
	{
		return (int_type)(round(a));
	}
	inline int_type myround(float a)
	{
		return (int_type)(roundf(a));
	}
	inline int_type myround(long double a)
	{
		return (int_type)(roundl(a));
	}

	inline void _update_pr()
	{
		// ensure we're always looking for something smaller than the first basis vector
		const double frac = 1.0 - (1.0/1024.0);
		pr[0] = std::min<float_type>(pr[0], risq[0]*frac);
		// ensure that the pruning bounds are non-increasing from a basis perspective.
		for (size_t k = 1; k < N; ++k)
			pr[k] = std::min<float_type>(pr[k-1],pr[k]);
	}

	// compile time parameters for enumerate_recur (without ANY runtime overhead)
	// allows specialization for certain specific cases, e.g., i=0
	template<int i, bool svp> struct i_tag {};

	template<int i, bool svp>
	inline void enumerate_recur(i_tag<i, svp>)
	{
		if (_r[i] > _r[i - 1])
			_r[i - 1] = _r[i];
		float_type ci = _sigT[i][i];
		float_type yi = round(ci);
		int_type xi = (int_type)(yi);
		yi = ci - yi;
		float_type li = _l[i + 1] + (yi * yi * risq[i]);
#ifndef NOCOUNTS
		++_counts[i];
#endif

		if (findsubsols && li < _subsolL[i] && li != 0.0)
		{
			_subsolL[i] = li;
			_subsol[i][i] = xi;
			for (int j = i + 1; j < N; ++j)
				_subsol[i][j] = _x[j];
		}
		if (li > pr[i])
			return;

		_Dx[i] = _D2x[i] = (((int)(yi >= 0) & 1) << 1) - 1;
		_c[i] = ci;
		_x[i] = xi;
		_l[i] = li;


		for (int j = _r[i - 1]; j > i - 1; --j)
			_sigT[i - 1][j - 1] = _sigT[i - 1][j] - _x[j] * muT[i - 1][j];

		while (true)
		{
			enumerate_recur(i_tag<i - 1, svp>());

			if (_l[i + 1] == 0.0)
			{
				++_x[i];
				_r[i - 1] = i;
				float_type yi2 = _c[i] - _x[i];
				float_type li2 = _l[i + 1] + (yi2 * yi2 * risq[i]);
				if (li2 > pr[i])
					return;
				_l[i] = li2;
				_sigT[i - 1][i - 1] = _sigT[i - 1][i] - _x[i] * muT[i - 1][i];
			}
			else
			{
				_x[i] += _Dx[i]; _D2x[i] = -_D2x[i]; _Dx[i] = _D2x[i] - _Dx[i];
				_r[i - 1] = i;
				float_type yi2 = _c[i] - _x[i];
				float_type li2 = _l[i + 1] + (yi2 * yi2 * risq[i]);
				if (li2 > pr[i])
					return;
				_l[i] = li2;
				_sigT[i - 1][i - 1] = _sigT[i - 1][i] - _x[i] * muT[i - 1][i];
			}
		}
	}

	template<bool svp>
	inline void enumerate_recur(i_tag<0, svp>)
	{
		static const int i = 0;
		float_type ci = _sigT[i][i];
		float_type yi = round(ci);
		int_type xi = (int_type)(yi);
		yi = ci - yi;
		float_type li = _l[i + 1] + (yi * yi * risq[i]);
#ifndef NOCOUNTS
		++_counts[i];
#endif

		if (findsubsols && li < _subsolL[i] && li != 0.0)
		{
			_subsolL[i] = li;
			_subsol[i][i] = xi;
			for (int j = i + 1; j < N; ++j)
				_subsol[i][j] = _x[j];
		}
		if (li > pr[i])
			return;

		_Dx[i] = _D2x[i] = (((int)(yi >= 0) & 1) << 1) - 1;
		_c[i] = ci;
		_x[i] = xi;
		_l[i] = li;


		while (true)
		{
			enumerate_recur(i_tag<i - 1, svp>());

			if (_l[i + 1] == 0.0)
			{
				++_x[i];
				float_type yi2 = _c[i] - _x[i];
				float_type li2 = _l[i + 1] + (yi2 * yi2 * risq[i]);
				if (li2 > pr[i])
					return;
				_l[i] = li2;
			}
			else
			{
				_x[i] += _Dx[i]; _D2x[i] = -_D2x[i]; _Dx[i] = _D2x[i] - _Dx[i];
				float_type yi2 = _c[i] - _x[i];
				float_type li2 = _l[i + 1] + (yi2 * yi2 * risq[i]);
				if (li2 > pr[i])
					return;
				_l[i] = li2;
			}
		}
	}


	template<bool svp>
	inline void enumerate_recur(i_tag<-1, svp>)
	{
		if (_l[0] > pr[0] || _l[0] == 0.0)
			return;

		for (int j = 0; j < N; ++j)
			_sol[j] = _x[j];

		pr[0] = _l[0];
		_update_pr();
	}

	template<bool svp = true>
	void enumerate_recursive()
	{
		_update_pr();

		for (int j = 0; j < N; ++j)
		{
			_x[j] = _Dx[j] = 0; _D2x[j] = -1;
			_sol[j] = 0;
			_c[j] = _l[j] = 0.0;
			_subsolL[j] = risq[j];
			for (int k = 0; k < N; ++k)
			{
				_sigT[j][k] = 0.0;
				_subsol[j][k] = 0;
			}
			_r[j] = N - 1;
			_counts[j] = 0;
		}
		_l[N] = 0.0;
		_counts[N] = 0;

		enumerate_recur(i_tag<N-1, svp>());

		std::cout << "[enum]: " << sqrt(pr[0]) << std::endl;
	}

};

#endif // ENUMLIB_ENUMERATION_HPP
