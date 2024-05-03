#include<algorithm>
#include<cmath>

#define FT double // floating-point type
#define ZZ long long // integer type

#define DECL(N) \
	void lll_reduce_ ## N(FT *R, ZZ *U, FT delta)
extern "C" {
	DECL( 1); DECL( 2); DECL( 3); DECL( 4); DECL( 5); DECL( 6); DECL( 7); DECL( 8);
	DECL( 9); DECL(10); DECL(11); DECL(12); DECL(13); DECL(14); DECL(15); DECL(16);
	DECL(17); DECL(18); DECL(19); DECL(20); DECL(21); DECL(22); DECL(23); DECL(24);
	DECL(25); DECL(26); DECL(27); DECL(28); DECL(29); DECL(30); DECL(31); DECL(32);
	DECL(33); DECL(34); DECL(35); DECL(36); DECL(37); DECL(38); DECL(39); DECL(40);
	DECL(41); DECL(42); DECL(43); DECL(44); DECL(45); DECL(46); DECL(47); DECL(48);
	DECL(49); DECL(50); DECL(51); DECL(52); DECL(53); DECL(54); DECL(55); DECL(56);
	DECL(57); DECL(58); DECL(59); DECL(60); DECL(61); DECL(62); DECL(63); DECL(64);
}

#define SQ(x) ((x) * (x))
#define RR(row, col) R[(row) * N + (col)]
#define UU(row, col) U[(row) * N + (col)]

template<const int N>
void lll_reduce(FT R[N * N], ZZ U[N * N], FT delta) {
	// Initialize the identity matrix
	std::fill_n(U, N * N, 0);
	for (int i = 0; i < N; i++) {
		UU(i, i) = 1;
	}

	// Assume R is an upper-triangular matrix.
	// Check if we can Lagrange reduce
	int k = 0;
	while (k < N - 1) {
		if (SQ(RR(k, k + 1)) + SQ(RR(k + 1, k + 1)) > delta * SQ(RR(k, k))) {
			// Already Lagrange reduced, move on.
			k++;
			continue;
		}

		// 1. Update U
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

#define FUNC(N) void lll_reduce_ ## N(FT *R, ZZ *U, FT delta) { lll_reduce<N>(R, U, delta); }

FUNC( 1) FUNC( 2) FUNC( 3) FUNC( 4) FUNC( 5) FUNC( 6) FUNC( 7) FUNC( 8)
FUNC( 9) FUNC(10) FUNC(11) FUNC(12) FUNC(13) FUNC(14) FUNC(15) FUNC(16)
FUNC(17) FUNC(18) FUNC(19) FUNC(20) FUNC(21) FUNC(22) FUNC(23) FUNC(24)
FUNC(25) FUNC(26) FUNC(27) FUNC(28) FUNC(29) FUNC(30) FUNC(31) FUNC(32)
FUNC(33) FUNC(34) FUNC(35) FUNC(36) FUNC(37) FUNC(38) FUNC(39) FUNC(40)
FUNC(41) FUNC(42) FUNC(43) FUNC(44) FUNC(45) FUNC(46) FUNC(47) FUNC(48)
FUNC(49) FUNC(50) FUNC(51) FUNC(52) FUNC(53) FUNC(54) FUNC(55) FUNC(56)
FUNC(57) FUNC(58) FUNC(59) FUNC(60) FUNC(61) FUNC(62) FUNC(63) FUNC(64)
