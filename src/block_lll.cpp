#include<algorithm>
#include<cmath>
#include<cstdio>

// floating-point type
typedef double FT;
// integer type
typedef long long ZZ;

extern "C" {
	bool lll_reduce(const int N, FT *R, ZZ *U, const FT delta, const size_t row_stride);
}

#define SQ(x) ((x) * (x))
#define RR(row, col) R[(row) * row_stride + (col)]
#define UU(row, col) U[(row) * row_stride + (col)]

/*
 * Applies size reduction to column j with respect to column i, and updates the
 * R-factor and transformation matrix U accordingly.
 */
inline
void size_reduce(const int N, FT *R, ZZ *U, size_t row_stride, int i, int j)
{
	ZZ quotient = llround(RR(i, j) / RR(i, i));
	if (quotient == 0) return;

	// R_j -= quotient * R_i.
	for (int k = 0; k <= i; k++) {
		RR(k, j) -= quotient * RR(k, i);
	}
	// U_i -= quotient * U_i.
	for (int k = 0; k < N; k++) {
		UU(k, j) -= quotient * UU(k, i);
	}
}

/*
 * Apply LLL to the basis, and return the transformation matrix U such that RU is LLL-reduced.
 */
bool lll_reduce(const int N, FT *R, ZZ *U, const FT delta, const size_t row_stride)
{
	// Initialize U with the identity matrix
	for (int i = 0; i < N; i++) {
		UU(i, i) = 1;
	}

	// Check that R is an upper-triangular matrix.
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < i; j++) {
			if (RR(i, j) < -0.5 or RR(i, j) > 0.5) {
				fprintf(stderr, "ERROR: R is not an upper-triangular matrix!\n");
				fflush(stderr);
				return false;
			}
		}
	}

	// Loop invariant: [0, k] is LLL-reduced (size-reduced and Lagrange reduced).
	for (int k = 0; k < N - 1; ) {
		// 1. Size-reduce R[k+1, n) wrt R[k].
		for (int i = k; i >= 0; i--) {
			size_reduce(N, R, U, row_stride, i, k + 1);
		}

		// 2. Check ||pi(b_{k+1})||^2 > \delta ||pi(b_k)||^2,
		// i.e. whether pi(b_k), pi(b_{k+1}) is Lagrange reduced.
		if (SQ(RR(k, k + 1)) + SQ(RR(k + 1, k + 1)) > delta * SQ(RR(k, k))) {
			// Already Lagrange reduced, move on.
			k++;
			continue;
		}

		// 3. Perform Givens rotation on coordinates {k, k+1}, and update R.
		FT c = RR(k, k + 1), s = RR(k + 1, k + 1), norm = sqrt(c * c + s * s);
		c /= norm;
		s /= norm;

		RR(k, k + 1) = c * RR(k, k);
		RR(k + 1, k + 1) = s * RR(k, k);
		RR(k, k) = norm;

		for (int i = k + 2; i < N; i++) {
			FT new_value = c * RR(k, i) + s * RR(k + 1, i);
			RR(k + 1, i) = s * RR(k, i) - c * RR(k + 1, i);
			RR(k, i) = new_value;
		}

		// 3. Swap U_k and U_{k+1}.
		for (int i = 0; i < N; i++) {
			std::swap(UU(i, k), UU(i, k + 1));
		}

		// 4. Swap R_k and R_{k+1} (except the already processed 2x2 block).
		for (int i = 0; i < k; i++) {
			std::swap(RR(i, k), RR(i, k + 1));
		}

		// 5. Make sure b_k is in fact LLL-reduced, so decrease `k`.
		if (k > 0) k--;
	}
	return true;
}
