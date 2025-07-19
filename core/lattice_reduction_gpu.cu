#include <cuda_runtime.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <cmath>

// floating-point type
typedef double FT;
// integer type
typedef long long ZZ;

#define RR(row, col) R[(row) * N + (col)]
#define UU(row, col) U[(row) * N + (col)]

__device__ inline void alter_basis(const int N, FT *R, ZZ *U, int i, int j, ZZ number) {
    if (number == 0) return;
    // R_j += number * R_i
    for (int k = 0; k <= i; k++) {
        RR(k, j) += number * RR(k, i);
    }
    // U_j += number * U_i
    for (int k = 0; k < N; k++) {
        UU(k, j) += number * UU(k, i);
    }
}

__device__ inline void size_reduce(const int N, FT *R, ZZ *U, int i, int j) {
    ZZ t = (ZZ)round(-RR(i, j) / RR(i, i));
    alter_basis(N, R, U, i, j, t);
}

__device__ void swap_basis_vectors(const int N, FT *R, ZZ *U, int k) {
    FT c = RR(k, k+1), s = RR(k+1, k+1);
    FT norm = sqrt(c*c + s*s);
    c /= norm; s /= norm;
    RR(k, k+1) = c*RR(k, k);
    RR(k+1, k+1) = s*RR(k, k);
    RR(k, k) = norm;
    for (int i = k+2; i < N; i++) {
        FT tmp = c*RR(k, i) + s*RR(k+1, i);
        RR(k+1, i) = s*RR(k, i) - c*RR(k+1, i);
        RR(k, i) = tmp;
    }
    for (int i = 0; i < k; i++) {
        // swap RR(i,k) <-> RR(i,k+1)
        FT t = RR(i, k); RR(i, k) = RR(i, k+1); RR(i, k+1) = t;
    }
    for (int i = 0; i < N; i++) {
        ZZ t = UU(i, k); UU(i, k) = UU(i, k+1); UU(i, k+1) = t;
    }
}

__device__ void init_U(const int N, ZZ *U) {
    // identity
    thrust::fill_n(thrust::device, U, N*N, ZZ(0));
    for (int i = 0; i < N; i++) UU(i, i) = 1;
}

__device__ void _lll_reduce(const int N, FT *R, ZZ *U, const FT delta) {
    int k = 1;
    while (k < N) {
        for (int i = k-1; i >= 0; --i) size_reduce(N, R, U, i, k);
        FT lhs = delta * RR(k-1, k-1)*RR(k-1, k-1);
        FT rhs = RR(k-1, k)*RR(k-1, k) + RR(k, k)*RR(k, k);
        if (lhs <= rhs) {
            k++;
        } else {
            swap_basis_vectors(N, R, U, k-1);
            if (k > 1) k--;
        }
    }
}

__device__ void lll_reduce(const int N, FT *R, ZZ *U, const FT delta) {
    init_U(N, U);
    _lll_reduce(N, R, U, delta);
}

// Kernel wrapper
extern "C" __global__
void lll_kernel(int N, double* R, long long* U, double delta) {
    // un seul thread fait tout
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        lll_reduce(N, R, U, delta);
    }
}

// Interface hôte
extern "C" void gpu_lll_reduce(const int N, double* h_R, long long* h_U, const double delta) {
    size_t sz = N * N * sizeof(double);
    size_t szU = N * N * sizeof(long long);

    double* d_R;
    long long* d_U;
    cudaMalloc(&d_R, sz);
    cudaMalloc(&d_U, szU);

    cudaMemcpy(d_R, h_R, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, h_U, szU, cudaMemcpyHostToDevice);

    // on lance 1 block de 1 thread (vous pouvez élargir si besoin)
    lll_kernel<<<1,1>>>(N, d_R, d_U, delta);
    cudaDeviceSynchronize();

    cudaMemcpy(h_R, d_R, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_U, d_U, szU, cudaMemcpyDeviceToHost);

    cudaFree(d_R);
    cudaFree(d_U);
}
