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
			// R_i -= q * R_k.
			for (int j = 0; j <= k; j++) RR(j, i) -= quotient * RR(j, k);
			// U_i -= q * R_k.
			for (int j = 0; j < N; j++) UU(j, i) -= quotient * UU(j, k);
		}

		if (k > 0) k--;
	}
}

#define DECL(N) void lll_reduce_ ## N(FT *R, ZZ *U, FT delta);
#define FUNC(N) void lll_reduce_ ## N(FT *R, ZZ *U, FT delta) { lll_reduce<N>(R, U, delta); }

extern "C" {
	FUNC( 1) FUNC( 2) FUNC( 3) FUNC( 4) FUNC( 5) FUNC( 6) FUNC( 7) FUNC( 8)
	FUNC( 9) FUNC(10) FUNC(11) FUNC(12) FUNC(13) FUNC(14) FUNC(15) FUNC(16)
	FUNC(17) FUNC(18) FUNC(19) FUNC(20) FUNC(21) FUNC(22) FUNC(23) FUNC(24)
	FUNC(25) FUNC(26) FUNC(27) FUNC(28) FUNC(29) FUNC(30) FUNC(31) FUNC(32)
	FUNC(33) FUNC(34) FUNC(35) FUNC(36) FUNC(37) FUNC(38) FUNC(39) FUNC(40)
	FUNC(41) FUNC(42) FUNC(43) FUNC(44) FUNC(45) FUNC(46) FUNC(47) FUNC(48)
	FUNC(49) FUNC(50) FUNC(51) FUNC(52) FUNC(53) FUNC(54) FUNC(55) FUNC(56)
	FUNC(57) FUNC(58) FUNC(59) FUNC(60) FUNC(61) FUNC(62) FUNC(63) FUNC(64)
}
