#include<algorithm> // std::swap, std::fill_n
#include<cmath> // llround, sqrt

#include "enumeration.cpp"
#include "pruning_params.cpp"

extern "C" {
	/*
	 * Perform LLL reduction on the basis R, and return the transformation matrix U such that
	 * RU is LLL-reduced.
	 * @param R upper-triangular matrix representing the R-factor from QR decomposition of the basis.
	 * @param U transformation matrix that is assumed to be the zero matrix upon calling this function.
	 *
	 * Complexity: poly(N) (for a fixed delta < 1).
	 */
	void lll_reduce(const int N, FT *R, ZZ *U, const FT delta);

	/*
	 * Perform depth-DeepLLL reduction on the basis R, and return the transformation matrix U such that
	 * RU is depth-DeepLLL-reduced.
	 *
	 * @param R upper-triangular matrix representing the R-factor from QR decomposition of the basis.
	 * @param U transformation matrix that is assumed to be the zero matrix upon calling this function.
	 * @param depth maximum number of positions allowed for 'deep insertions'
	 *
	 * Complexity: poly(N) (for a fixed delta < 1 and a fixed depth).
	 */
	void deeplll_reduce(const int N, FT *R, ZZ *U, const FT delta, int depth);

	/*
	 * Perform BKZ-beta reduction on the basis R, and return the transformation matrix U such that
	 * RU is BKZ-beta-reduced.
	 *
	 * @param R upper-triangular matrix representing the R-factor from QR decomposition of the basis.
	 * @param U transformation matrix that is assumed to be the zero matrix upon calling this function.
	 * @param beta blocksize used for BKZ (dimension of SVP oracle that uses enumeration).
	 *
	 * Complexity: poly(N) * beta^{c_BKZ beta} for a fixed delta < 1, where c_BKZ ~ 0.125 in [2].
	 * [2] https://doi.org/10.1007/978-3-030-56880-1_7
	 */
	void bkz_reduce(const int N, FT *R, ZZ *U, const FT delta, int beta);

	/*
	 * Perform HKZ reduction on the basis R, and return the transformation matrix U such that
	 * RU is HKZ-reduced.
	 *
	 * @param R upper-triangular matrix representing the R-factor from QR decomposition of the basis.
	 * @param U transformation matrix that is assumed to be the zero matrix upon calling this function.
	 *
	 * Complexity: poly(N) * N^{c_BKZ N} for a fixed delta < 1, where c_BKZ ~ 0.125 in [2].
	 */
	void hkz_reduce(const int N, FT *R, ZZ *U, const FT delta)
	{
		bkz_reduce(N, R, U, delta, N);
	}
}


/*******************************************************************************
 * Helper functions to access the matrices R and U at row 'row' and column 'col'
 ******************************************************************************/
#define RR(row, col) R[(row) * N + (col)]
#define RSQ(row, col) (RR(row, col) * RR(row, col))

#define UU(row, col) U[(row) * N + (col)]

/*
 * Replace `b_j` by `b_j + number * b_i`, and
 * update R-factor and transformation matrix U accordingly.
 * Assumes i < j.
 *
 * Complexity: O(N)
 */
inline void alter_basis(const int N, FT *R, ZZ *U, int i, int j, ZZ number)
{
	if (number == 0) {
		return;
	}

	// R_j += number * R_i.
	for (int k = 0; k <= i; k++) {
		RR(k, j) += number * RR(k, i);
	}

	// U_i += number * U_i.
	for (int k = 0; k < N; k++) {
		UU(k, j) += number * UU(k, i);
	}
}
/*
 * Apply size reduction to column j with respect to column i, and
 * update the R-factor and transformation matrix U accordingly.
 *
 * Complexity: O(N)
 */
inline void size_reduce(const int N, FT *R, ZZ *U, int i, int j)
{
	ZZ quotient = llround(RR(i, j) / RR(i, i));
	alter_basis(N, R, U, i, j, -quotient);
}

/*
 * Swap the adjacent basis vectors b_k and b_{k+1} and update the R-factor and transformation
 * matrix U correspondingly. R is updated by performing a Givens rotation.
 *
 * Complexity: O(N)
 */
void swap_basis_vectors(const int N, FT *R, ZZ *U, const int k)
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

/*******************************************************************************
 * LLL reduction
 ******************************************************************************/
void _lll_reduce(const int N, FT *R, ZZ *U, const FT delta, const int limit_k)
{
	// Loop invariant: [0, k) is LLL-reduced (size-reduced and Lagrange reduced).
	for (int k = 1; k < N; ) {
		// 1. Size-reduce R_k wrt R_0, ..., R_{k - 1}.
		for (int i = k - 1; i >= 0; --i) {
			size_reduce(N, R, U, i, k);
		}

		// 2. Check ||pi(b_k)||^2 > \delta ||pi(b_{k - 1})||^2.
		if (RSQ(k - 1, k) + RSQ(k, k) > delta * RSQ(k - 1, k - 1)) {
			// pi(b_{k - 1}), pi(b_k) is already Lagrange reduced, so move on.
			k++;
		} else {
			// 3. Swap b_{k - 1} and b_k.
			swap_basis_vectors(N, R, U, k - 1);

			// 4. Decrease `k` if possible.
			if (k > 1) k--;
		}
	}
}

/*
 * Performs LLL reduction on b_0, ..., b_{limit_k - 1}.
 */
void lll_reduce(const int N, FT *R, ZZ *U, const FT delta)
{
	// Initialize U with the identity matrix
	std::fill_n(U, N * N, ZZ(0));
	for (int i = 0; i < N; i++) {
		UU(i, i) = 1;
	}

	_lll_reduce(N, R, U, delta, N);
}

/*******************************************************************************
 * LLL reduction with deep insertions
 ******************************************************************************/
void _deeplll_reduce(const int N, FT *R, ZZ *U, const FT delta, const int depth, const int limit_k)
{
	// Loop invariant: [0, k) is (depth-deep)LLL-reduced (size-reduced and Lagrange reduced).
	for (int k = 1; k < limit_k; ) {
		// 1. Size-reduce R_k wrt R_0, ..., R_{k - 1}.
		for (int i = k - 1; i >= 0; i--) {
			size_reduce(N, R, U, i, k);
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
					swap_basis_vectors(N, R, U, --k);
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

void deeplll_reduce(const int N, FT *R, ZZ *U, const FT delta, const int depth)
{
	// If 'depth' is not supplied, set it to N.
	// Note: first running 'standard' LLL, before performing deepLLL is much faster on average.
	// [1] https://doi.org/10.1007/s10623-014-9918-8
	lll_reduce(N, R, U, delta);

	_deeplll_reduce(N, R, U, delta, depth > 0 ? depth : N, N);
}

/*******************************************************************************
 * BKZ reduction
 ******************************************************************************/

/*
 * Compute and return the square of the Gaussian Heuristic, i.e. (the square
 * of) the expected length of the shortest nonzero vector, for a lattice of
 * dimension `dimension` with a determinant (covolume) of `exp(log_volume)`.
 *
 * @param dimension dimension of lattice
 * @param log_volume logarithm of volume of the lattice, natural base.
 * @return GH(L)^2, for a lattice L of rank `dimension` and `det(L) = exp(log_volume)`.
 */
FT gh_squared(int dimension, FT log_volume)
{
	// GH(n) = Gamma(n / 2 + 1) / (pi)^{n / 2} so
	// GH(n)^2 = exp(log(Gamma(n / 2 + 1)) / n) / pi.
	FT log_gamma = lgamma(dimension / 2.0 + 1);
	return exp(2.0 * (log_gamma + log_volume) / dimension) / M_PI;
}

/*
 * Solve SVP on b_[i, i + w), and
 * put the result somewhere in the basis where the coefficient is +1/-1, and
 * run LLL on b_1, ..., b_{i + w}.
 */
bool svp(const int N, FT *R, ZZ *U, const FT delta, int i, int w, ZZ *sol)
{
	int h = i + w;
	if (h < N) h++;

	FT log_volume = 0.0;
	for (int j = 0; j < w; j++)
		log_volume += log(RSQ(i + j, i + j));

	// Find a solution that is shorter than current basis vector and of norm <1.2 GH (<2.0 GH for dim < 10)
	FT expected_normsq = std::min((1023.0 / 1024) * RSQ(i, i), (w < 10 ? 4.0 : 1.44) * gh_squared(w, log_volume));

	// 1. Pick the pruning parameters for `pr[0 ... w - 1]`.
	const FT *pr = get_pruning_coefficients(w);

	// 2. Enumerate.
	// [3] Algorithm 1, line 4
	FT sol_square_norm = enumeration(w, &RR(i, i), N, pr, expected_normsq, sol);

	// 3. Check if it returns a shorter & nonzero vector.
	// [3] Algorithm 1, line 5
	if (sol_square_norm == 0.0) {
		// No better solution was found, because:
		// a. pruning caused to miss a shorter vector (prob. ~ 1%), or
		// b. b_i is already the shortest vector in the block [i, j).

		// [3] Algorithm 1, line 8:
		// LLL-reduce the next block before enumeration.
		_deeplll_reduce(N, R, U, delta, 4, h);
		return false;
	}

	// 4. Find the last +1/-1 coefficient.
	int insert_idx = w - 1;
	while (insert_idx >= 0 && sol[insert_idx] != 1 && sol[insert_idx] != -1) {
		insert_idx--;
	}

	if (insert_idx < 0) {
		// This should not happen so regularly!
		// We should always have gcd(sol) = 1, but there can be no `i` with `sol[i] = +1/-1`!
		// TODO: report this as an error?!
		return false;
	}

	// 5. Replace the solution by its opposite, if sol[insert_idx] = -1.
	if (sol[insert_idx] == -1) {
		for (int j = 0; j <= insert_idx; j++) {
			sol[j] = -sol[j];
		}
	}

	// 6. Update for all 0 <= j < insert_idx:
	for (int j = 0; j < insert_idx; j++) {
		// b_{i + insert_idx} += sol[j] * b_{i + j}.
		alter_basis(N, R, U, i + j, i + insert_idx, sol[j]);
	}

	// 7. Move b_{i + insert_idx} to position b_i.
	while (--insert_idx >= 0) {
		swap_basis_vectors(N, R, U, i + insert_idx);
	}

	// 8. Run LLL on [0, h).
	// [3] Algorithm 1, line 6
	_deeplll_reduce(N, R, U, delta, 4, h);

	return true;
}

void bkz_reduce(const int N, FT *R, ZZ *U, const FT delta, int beta)
{
	ZZ sol[MAX_ENUM_N]; // coefficients of the enumeration solution for SVP in block of size beta.

	// First run 'standard' LLL, before performing BKZ.
	lll_reduce(N, R, U, delta);

	if (beta <= 2) return;

	if (beta > N) {
		// Perform one HKZ-tour.
		// In the global picture, we only do this at the end of the basis!
		for (int i = 0; i + 2 <= N; i++) {
			// Solve SVP on block [i, N).
			svp(N, R, U, delta, i, N - i, sol);
		}
	} else {
		// Perform one BKZ-tour.
		for (int i = 0, w = beta; i + w <= N; i++) {
			// Solve SVP on block [i, i + w).
			svp(N, R, U, delta, i, w, sol);
		}
	}
}
