#include<algorithm>
#include<cmath>

#include<cstdio>

#define FT double // floating-point type
#define ZZ long long // integer type

#define SQ(x) ((x) * (x))
#define RR(row, col) R[(row) * N + (col)]
#define UU(row, col) U[(row) * N + (col)]

template<const int N>
void lll_reduce(FT R[N * N], ZZ U[N * N], FT delta) {
	// Initialize U with the identity matrix
	std::fill_n(U, N * N, 0);
	for (int i = 0; i < N; i++) {
		UU(i, i) = 1;
	}

	// Check that R is an upper-triangular matrix.
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < i; j++) {
			if (RR(i, j) < -0.5 or RR(i, j) > 0.5) {
				fprintf(stderr, "ERROR: R is not an upper-triangular matrix!\n");
				fflush(stderr);
				return;
			}
		}
	}

	// Loop invariant: [0, k] is LLL-reduced (size-reduced and Lagrange reduced).
	int k = 0;
	int iters = 0;
	while (k < N - 1) {
		if (++iters > 10'000'000) {
			fprintf(stderr, "ERROR: a potentially infinite loop is detected in LLL!\n");
			fflush(stderr);
			break;
		}

		if (SQ(RR(k, k + 1)) + SQ(RR(k + 1, k + 1)) > delta * SQ(RR(k, k))) {
			// Already Lagrange reduced, move on.
			k++;
			continue;
		}

		// 1. Update U: swap k-th and (k+1)-th column of U.
		for (int i = 0; i < N; i++) {
			std::swap(UU(i, k), UU(i, k + 1));
		}

		// 2. Swap R_k and R_k+1 & perform Givens rotation on R[k:n)
		FT c = RR(k, k + 1), s = RR(k + 1, k + 1); // resp. cos & sin of rotation
		FT norm = sqrt(c * c + s * s);
		c /= norm; s /= norm;

		RR(k, k + 1) = c * RR(k, k);
		RR(k + 1, k + 1) = s * RR(k, k);
		RR(k, k) = norm;

		for (int i = k + 2; i < N; i++) {
			FT new_value = c * RR(k, i) + s * RR(k + 1, i);
			RR(k + 1, i) = s * RR(k, i) - c * RR(k + 1, i);
			RR(k, i) = new_value;
		}

		// 3. Also size-reduce R[k+1, n) wrt R[k].
		for (int i = k + 1; i < N; i++) {
			ZZ quotient = llround(RR(k, i) / norm);
			if (quotient == 0) continue;
			// R_i -= quotient * R_k.
			for (int j = 0; j <= k; j++) {
				RR(j, i) -= quotient * RR(j, k);
			}
			// U_i -= quotient * R_k.
			for (int j = 0; j < N; j++) {
				UU(j, i) -= quotient * UU(j, k);
			}
		}

		if (k > 0) k--;
	}
}

#define F(N) \
	void lll_reduce_##N(FT *R, ZZ *U, FT delta) { lll_reduce<N>(R, U, delta); }

extern "C" {
	F( 1) F( 2) F( 3) F( 4) F( 5) F( 6) F( 7) F( 8)
	F( 9) F(10) F(11) F(12) F(13) F(14) F(15) F(16)
	F(17) F(18) F(19) F(20) F(21) F(22) F(23) F(24)
	F(25) F(26) F(27) F(28) F(29) F(30) F(31) F(32)
	F(33) F(34) F(35) F(36) F(37) F(38) F(39) F(40)
	F(41) F(42) F(43) F(44) F(45) F(46) F(47) F(48)
	F(49) F(50) F(51) F(52) F(53) F(54) F(55) F(56)
	F(57) F(58) F(59) F(60) F(61) F(62) F(63) F(64)
}
