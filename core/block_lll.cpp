#include<algorithm> // std::swap, std::fill_n
#include<cmath> // llround, sqrt

#include<cstdio>

/*
 * Import local code to perform enumeration to do BKZ in smallish blocksizes.
 */
#include "enumeration.cpp"

constexpr int MAX_ENUM_N = 256; // See enumeration.cpp:16


// floating-point type
typedef double FT;

// integer type
typedef long long ZZ;


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
	 * @param max_tours upper bound on number of BKZ tours performed, or disabled when <= 0.
	 *
	 * Complexity: poly(N) * beta^{c_BKZ beta} for a fixed delta < 1, where c_BKZ ~ 0.125 in [2].
	 * [2] https://doi.org/10.1007/978-3-030-56880-1_7
	 */
	void bkz_reduce(const int N, FT *R, ZZ *U, const FT delta, int beta, int max_tours);

	/*
	 * Perform HKZ reduction on the basis R, and return the transformation matrix U such that
	 * RU is HKZ-reduced.
	 *
	 * @param R upper-triangular matrix representing the R-factor from QR decomposition of the basis.
	 * @param U transformation matrix that is assumed to be the zero matrix upon calling this function.
	 * @param max_tours upper bound on number of BKZ tours performed, or disabled when <= 0.
	 *
	 * Complexity: poly(N) * N^{c_BKZ N} for a fixed delta < 1, where c_BKZ ~ 0.125 in [2].
	 */
	void hkz_reduce(const int N, FT *R, ZZ *U, const FT delta, int max_tours);
}


/*
 * Helper functions to access the matrices R and U at row 'row' and column 'col'
 */
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
inline
void alter_basis(const int N, FT *R, ZZ *U, int i, int j, ZZ number)
{
	if (number == 0) {
		return;
	}

	// R_j -= number * R_i.
	for (int k = 0; k <= i; k++) {
		RR(k, j) -= number * RR(k, i);
	}

	// U_i -= number * U_i.
	for (int k = 0; k < N; k++) {
		UU(k, j) -= number * UU(k, i);
	}
}
/*
 * Applies size reduction to column j with respect to column i, and
 * update the R-factor and transformation matrix U accordingly.
 *
 * Complexity: O(N)
 */
inline
void size_reduce(const int N, FT *R, ZZ *U, int i, int j)
{
	ZZ quotient = llround(RR(i, j) / RR(i, i));
	alter_basis(N, R, U, i, j, quotient);
}

/*
 * Swap the adjacent basis vectors b_k and b_{k+1} and update the R-factor and transformation
 * matrix U correspondingly. R is updated by performing a Givens rotation.
 *
 * Complexity: O(N)
 */
inline
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


/*
 * Warning: this does not initialize U to identity matrix.
 * Performs LLL reduction on b_0, ..., b_{limit_k - 1}.
 */
void internal_lll_reduce(const int N, FT *R, ZZ *U, const FT delta, const int limit_k)
{
	// Loop invariant: [0, k) is LLL-reduced (size-reduced and Lagrange reduced).
	for (int k = 1; k < limit_k; ) {
		// 1. Size-reduce R_k wrt R_0, ..., R_{k - 1}.
		for (int i = k - 1; i >= 0; --i) {
			size_reduce(N, R, U, i, k);
		}

		// 2. Check ||pi(b_k)||^2 > \delta ||pi(b_{k - 1})||^2.
		if (RSQ(k - 1, k) + RSQ(k, k) > delta * RSQ(k - 1, k - 1)) {
			// pi(b_{k - 1}), pi(b_k) is Lagrange reduced, so move on.
			// Already Lagrange reduced, move on.
			k++;
		} else {
			// 3. Swap b_{k - 1} and b_k.
			swap_basis_vectors(N, R, U, k - 1);

			// 4. Decrease `k` if possible.
			if (k > 1) k--;
		}
	}

	// TODO: Remove debugging information.
/*
	printf("Basis norms: ");
	for (int i = 0; i < limit_k; i++) {
		FT normsq = 0.0;
		for (int j = 0; j <= i; j++) normsq += RSQ(j, i);
		printf("%6.2f ", normsq);
	}
	printf("; R-matrix:\n");
	for (int i = 0; i < limit_k; i++) {
		for (int j = 0; j < limit_k; j++) {
			if (j < i) printf("       ");
			else printf("%6.2f ", RR(i, j));
		}
		printf("\n");
	}
*/

}

void lll_reduce(const int N, FT *R, ZZ *U, const FT delta)
{
	// Initialize U with the identity matrix
	std::fill_n(U, N * N, ZZ(0));
	for (int i = 0; i < N; i++) {
		UU(i, i) = 1;
	}

	// Call internal function
	internal_lll_reduce(N, R, U, delta, N);
}

void deeplll_reduce(const int N, FT *R, ZZ *U, const FT delta, int depth)
{
	// If 'depth' is not supplied, set it to N.
	if (depth <= 0) {
		depth = N;
	}

	// Note: first running 'standard' LLL, before performing deepLLL is much faster on average.
	// [1] https://doi.org/10.1007/s10623-014-9918-8
	lll_reduce(N, R, U, delta);

	// Loop invariant: [0, k) is (depth-deep)LLL-reduced (size-reduced and Lagrange reduced).
	for (int k = 1; k < N; ) {
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

/*
 * Solve SVP on b_[i, i + w), and
 * put the result somewhere in the basis where the coefficient is +1/-1, and
 * run LLL on b_1, ..., b_{i + w}.
 */
bool internal_svp(const int N, FT *R, ZZ *U, const FT delta, int i, int w, ZZ *sol, FT *pr)
{
	int h = i + w;
	if (h < N) h++;

	// 1. Pick the pruning parameters for `pr[0 ... w - 1]`.
	for (int j = 0; j < w; j++) {
		// TODO: find stronger pruning parameters (with ~99% success probability) based on [3].
		// [3] https://doi.org/10.1007/978-3-642-25385-0_1
		FT pr_j = 1.0;

		pr[j] = pr_j * RSQ(i, i);
	}

	// 2. Enumerate.
	// [3] Algorithm 1, line 4
	FT sol_square_norm = enumeration(w, &RR(i, i), N, pr, sol);

	// 3. Check if it returns a shorter & nonzero vector.
	// [3] Algorithm 1, line 5
	// Note this might have 2 reasons:
	//   a. pruning caused to miss a shorter vector (prob. ~ 1%), or
	//   b. b_i is already the shortest vector in the block [i, j).
	if (sol_square_norm == 0.0) {
		// We are in case 3a. or 3b.

		// [3] Algorithm 1, line 8:
		// LLL-reduce the next block before enumeration.
		internal_lll_reduce(N, R, U, delta, h);
		return false;
	}

	// TODO: remove debugging information.
/*
	printf("(%d, %d, %d) -> Original basis vector lengths: %.3f %.3f %.3f\n",
			N, i, w,
			RSQ(i, i),
			RSQ(i + 1, i + 1) + RSQ(i, i + 1),
			RSQ(i + 2, i + 2) + RSQ(i + 1, i + 2) + RSQ(i, i + 2));

	printf("Enumeration found norm^2 %.3f, solution: ", sol_square_norm);
	for (int j = 0; j < w; j++) printf("%d ", sol[j]);
	printf("\n");
*/

	// 4. Find the last +1/-1 coefficient.
	int insert_idx = w - 1;
	while (insert_idx >= 0 && sol[insert_idx] != 1 && sol[insert_idx] != -1) {
		insert_idx--;
	}

	if (insert_idx < 0) {
		// This should not happen so regularly!
		// We should always have gcd(sol) = 1, but it's not guaranteed there is an `i` with `sol[i] = +1/-1`...
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
	internal_lll_reduce(N, R, U, delta, h);

	return true;
}

void bkz_reduce(const int N, FT *R, ZZ *U, const FT delta, int beta, int max_tours)
{
	ZZ sol[MAX_ENUM_N]; // coefficients of the enumeration solution for SVP in block of size beta.
	FT pr[MAX_ENUM_N]; // pruning parameters

	// First run 'standard' LLL, before performing BKZ.
	lll_reduce(N, R, U, delta);

	// If 'depth' is not supplied, set it to N.
	if (beta <= 2) {
		return;
	}

	if (beta > N) {
		// Perform HKZ-reduction
		beta = N;
	}

	if (max_tours <= 0) {
		max_tours = N;
		// max_tours = INT_MAX;
	}

	bool changed = true;
	while (max_tours-- > 0 && changed) {
		changed = false;

		// Perform a tour.
		// Solve SVP in blocks
		// [0, beta), [1, 1 + beta), ..., [N - beta, N), [N - beta + 1, N), ..., [N - 2, N).
		for (int i = 0, w = beta; i + 2 <= N; i++) {
			changed |= internal_svp(N, R, U, delta, i, w, sol, pr);

			// Decrease the blocksize once we are at the end, because that part is HKZ-reduced.
			if (i + w == N) w--;
		}
	}
}

void hkz_reduce(const int N, FT *R, ZZ *U, const FT delta, int max_tours)
{
	bkz_reduce(N, R, U, delta, N, max_tours);
}
