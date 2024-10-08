#include<algorithm> // std::swap
#include<cmath> // llround, sqrt

// floating-point type
typedef double FT;

// integer type
typedef long long ZZ;


extern "C" {
	void lll_reduce(const int N, FT *R, ZZ *U, const FT delta, const size_t row_stride);
	void deeplll_reduce(const int N, FT *R, ZZ *U, const FT delta,
			const size_t row_stride, int depth);
}


/*
 * Helper functions to access the matrices R and U at row 'row' and column 'col'
 */
#define RR(row, col) R[(row) * row_stride + (col)]
#define RSQ(row, col) (RR(row, col) * RR(row, col))

#define UU(row, col) U[(row) * N + (col)]

/*
 * Applies size reduction to column j with respect to column i, and updates the R-factor and
 * transformation matrix U accordingly.
 *
 * Complexity: O(N)
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
 * Swaps the adjacent basis vectors b_k and b_{k+1} and updates the R-factor and transformation
 * matrix U correspondingly. R is updated by performing a Givens rotation.
 *
 * Complexity: O(N)
 */
inline
void swap_basis_vectors(const int N, FT *R, ZZ *U, const size_t row_stride, const int k)
{
	// a. Perform Givens rotation on coordinates {k, k+1}, and update R.
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

	// b. Swap R_k and R_{k+1}, except the already processed 2x2 block.
	for (int i = 0; i < k; i++) {
		std::swap(RR(i, k), RR(i, k + 1));
	}

	// c. Swap U_k and U_{k+1}.
	for (int i = 0; i < N; i++) {
		std::swap(UU(i, k), UU(i, k + 1));
	}

}

/*
 * Apply LLL to the basis, and return the transformation matrix U such that RU is LLL-reduced.
 * @param R upper-triangular matrix representing the R-factor from QR decomposition of the basis.
 * @param U transformation matrix that is assumed to be the zero matrix upon calling this function.
 *
 * Complexity: poly(N) (for a fixed delta < 1).
 */
void lll_reduce(const int N, FT *R, ZZ *U, const FT delta, const size_t row_stride)
{
	std::fill_n(U, N * N, ZZ(0));
	// Initialize U with the identity matrix
	for (int i = 0; i < N; i++) {
		UU(i, i) = 1;
	}

	// Loop invariant: [0, k) is LLL-reduced (size-reduced and Lagrange reduced).
	for (int k = 1; k < N; ) {
		// 1. Size-reduce R_k wrt R_0, ..., R_{k - 1}.
		for (int i = k - 1; i >= 0; --i) {
			size_reduce(N, R, U, row_stride, i, k);
		}

		// 2. Check ||pi(b_k)||^2 > \delta ||pi(b_{k - 1})||^2.
		if (RSQ(k - 1, k) + RSQ(k, k) > delta * RSQ(k - 1, k - 1)) {
			// pi(b_{k - 1}), pi(b_k) is Lagrange reduced, so move on.
			// Already Lagrange reduced, move on.
			k++;
		} else {
			// 3. Swap b_{k - 1} and b_k.
			swap_basis_vectors(N, R, U, row_stride, k - 1);

			// 4. Decrease `k` if possible.
			if (k > 1) k--;
		}
	}
}

/*
 * Apply DeepLLL to the basis, and return the transformation matrix U such that
 * RU is LLL-reduced.
 *
 * @param R upper-triangular matrix representing the R-factor from QR decomposition of the basis.
 * @param U transformation matrix that is assumed to be the zero matrix upon calling this function.
 * @param depth maximum number of positions allowed for 'deep insertions'
 *
 * Complexity: poly(N) (for a fixed delta < 1 and a fixed depth).
 */
void deeplll_reduce(const int N, FT *R, ZZ *U, const FT delta,
		const size_t row_stride, int depth)
{
	// If 'depth' is not supplied, set it to N.
	if (depth <= 0) {
		depth = N;
	}

	// Note: first running 'standard' LLL, before performing deepLLL is much faster on average.
	// [1] https://doi.org/10.1007/s10623-014-9918-8
	lll_reduce(N, R, U, delta, row_stride);

	// Loop invariant: [0, k) is (depth-deep)LLL-reduced (size-reduced and Lagrange reduced).
	for (int k = 1; k < N; ) {
		// 1. Size-reduce R_k wrt R_0, ..., R_{k - 1}.
		for (int i = k - 1; i >= 0; i--) {
			size_reduce(N, R, U, row_stride, i, k);
		}

		// 2. Determine ||b_k||^2
		FT proj_norm_sq = 0.0;
		for (int i = 0; i <= k; i++) {
			proj_norm_sq += RSQ(i, k);
		}

		// 3. Look for an i < k such that ||pi_i(b_k)||^2 <= delta ||b_i*||^2.
		// Loop invariant: proj_norm_sq = ||pi_i(b_k)||^2.
		bool swap_performed = false;
		for (int i = 0; i < k; i++) {
			if ((i < depth || i >= k - depth) && proj_norm_sq <= delta * RSQ(i, i)) {
				// 3a. Deep insert b_k at position i and move b_i, ..., b_{k-1}
				// one position forward. Complexity: O(N * (k - i))
				while (k > i) {
					swap_basis_vectors(N, R, U, row_stride, --k);
				}
				if (k == 0) k++;
				swap_performed = true;
				break;
			}

			// 3b. Increment i and update ||pi_i(b_k)||^2.
			proj_norm_sq -= RSQ(i, k);
		}

		if (!swap_performed) k++;
	}
}
