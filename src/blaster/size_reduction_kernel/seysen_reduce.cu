#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <unordered_map>
#include <mutex>
#include <tuple>
//------------------------------------------------------------------------------
// Macros pour la vérification d'erreurs
//------------------------------------------------------------------------------



#ifdef DEBUG_DUMP_MAT

#define DEBUG_HOST(msg) \
  fprintf(stderr, "[HOST] %s:%d %s\n", __FILE__, __LINE__, msg);

#define CHECK_LAST()                                                    \
  do {                                                                  \
    cudaError_t e = cudaGetLastError();                                 \
    if (e != cudaSuccess)                                               \
      fprintf(stderr, "[CUDAERR] %s:%d → %s\n",                         \
              __FILE__, __LINE__, cudaGetErrorString(e));               \
  } while(0)


#define CUDA_CHECK(call)                                                      \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                           \
  } while (0)

#define CUBLAS_CHECK(call)                                                      \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "%s:%d CUBLAS error: %d\n", __FILE__, __LINE__,       \
              status);                                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                           \
  } while (0)


#define WRITE_MAT(fname, d_ptr, rows, cols, ld)                                 \
    do {                                                                        \
        std::vector<double> _hbuf((rows) * (cols));                             \
        CUDA_CHECK(cudaMemcpy2D(                                                \
            _hbuf.data(),                                                        \
            (cols) * sizeof(double),                                            \
            (d_ptr),                                                             \
            (ld) * sizeof(double),                                              \
            (cols) * sizeof(double),                                            \
            (rows),                                                              \
            cudaMemcpyDeviceToHost                                              \
        ));                                                                      \
        write_matrix_csv_c((fname), _hbuf.data(), (rows), (cols));              \
    } while (0)

#define WRITE_MAT_3D(base, d_ptr, batch, rows, cols, ld)                         \
    do {                                                                        \
        /* Tampon hôte pour une tranche */                                      \
        std::vector<double> _hbuf((rows) * (cols));                             \
        for (int bi = 0; bi < (batch); ++bi) {                                  \
            /* Construction du nom de fichier pour la tranche bi */             \
            char _fname[256];                                                   \
            std::snprintf(_fname, sizeof(_fname), "%s_batch_%d.csv", (base), bi);\
                                                                                \
            /* Copie 2D Device→Host de la tranche bi : */                       \
            /* src = d_ptr + bi*rows*ld (début de la tranche en devicemem) */     \
            CUDA_CHECK(cudaMemcpy2D(                                            \
                _hbuf.data(),                      /* dst */                    \
                (cols) * sizeof(double),           /* dpitch */                \
                (d_ptr) + size_t(bi) * (rows) * (ld), /* src */                 \
                (ld) * sizeof(double),              /* spitch */               \
                (cols) * sizeof(double),           /* width */                 \
                (rows),                             /* height */                \
                cudaMemcpyDeviceToHost                                              \
            ));                                                                  \
                                                                                \
            /* Écriture CSV de la tranche */                                     \
            write_matrix_csv_c(_fname, _hbuf.data(), (rows), (cols));           \
        }                                                                       \
    } while (0)

#endif

static void write_matrix_csv_c(char* filename, double* M, int h, int w) {
    FILE* f = std::fopen(filename, "w");
    if (!f) { std::perror(filename); return; }
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            std::fprintf(f, "%g", M[i*w + j]);
            if (j < w-1) std::fprintf(f, ",");
        }
        std::fprintf(f, "\n");
    }
    std::fclose(f);
}

static void write_matrix_csv_c(const char* filename, const int64_t* M, int n) {
    FILE* f = std::fopen(filename, "w");
    if (!f) { std::perror(filename); return; }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::fprintf(f, "%lld", (long long)M[i*n + j]);
            if (j < n-1) std::fprintf(f, ",");
        }
        std::fprintf(f, "\n");
    }
    std::fclose(f);
}

//------------------------------------------------------------------------------
// Structure décrivant le plan de réduction
//------------------------------------------------------------------------------
struct ReductionPlan {
    int n;               // dimension de la matrice
    int m;               // nombre total de ranges
    int min_batch;       // batch minimum à traiter en une passe
};

//------------------------------------------------------------------------------
// Buffers pour les opérations batched
//------------------------------------------------------------------------------

struct BatchBuffers {
    double *storage;
    double *d_R11, *d_R12, *d_U22, *d_U11;
    double *d_S12, *d_S12_old;
    double *d_U12;
    int batch_size, w;
};



void allocateBatchBuffers(BatchBuffers &buf, int batch_size, int w) {
    buf.batch_size = batch_size;
    buf.w = w;
    size_t bytes = size_t(batch_size) * w * w * sizeof(double);
    int nb_arrays = 7; // d_R11, d_R12, d_U22, d_S12, d_U11, d_S12_old, d_U12

    // 1 seul gros buffer
    (cudaMalloc(&buf.storage, nb_arrays * bytes));

    // Et maintenant on découpe :
    double* base = static_cast<double*>(buf.storage);
    buf.d_R11     = base;
    buf.d_R12     = base + 1 * batch_size * w * w;
    buf.d_U22     = base + 2 * batch_size * w * w;
    buf.d_S12     = base + 3 * batch_size * w * w;
    buf.d_U11     = base + 4 * batch_size * w * w;
    buf.d_S12_old = base + 5 * batch_size * w * w;
    buf.d_U12     = base + 6 * batch_size * w * w;
}



void freeBatchBuffers(BatchBuffers &buf) {
    cudaFree(buf.d_R11);
    cudaFree(buf.d_R12);
    cudaFree(buf.d_U22);
    cudaFree(buf.d_S12);
    cudaFree(buf.d_U11);
    cudaFree(buf.d_S12_old);
    cudaFree(buf.d_U12);
}

struct CachedBatchBuffer {
    BatchBuffers buf;
    int last_m = -1;
    bool is_allocated = false;
} g_cached_batch_buf;



BatchBuffers* getOrAllocBatchBuffer(int m, int batch_size, int w) {
    if (g_cached_batch_buf.is_allocated && g_cached_batch_buf.last_m == m) {
        // On garde le buffer existant !
        return &g_cached_batch_buf.buf;
    }
    // Sinon, faut free et re-alloc
    if (g_cached_batch_buf.is_allocated) {
        freeBatchBuffers(g_cached_batch_buf.buf);
    }
    allocateBatchBuffers(g_cached_batch_buf.buf, batch_size, w);
    g_cached_batch_buf.last_m = m;
    g_cached_batch_buf.is_allocated = true;
    return &g_cached_batch_buf.buf;
}

void free_cached_batch_buffer() {
    if (g_cached_batch_buf.is_allocated) {
        freeBatchBuffers(g_cached_batch_buf.buf);
        g_cached_batch_buf.is_allocated = false;
        g_cached_batch_buf.last_m = -1;
    }
}

//------------------------------------------------------------------------------
// Kernels nommés selon chaque étape
//------------------------------------------------------------------------------
// Étape 1: Processus des "base cases"
// Étape 1: Processus des "base cases" (utilise d_base_cases constant)
__global__ void kernelProcessBaseCases(
    double* R, double* U,
    int b, const int*     base_cases, int n, int ldu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= b) return;
    int i = base_cases[idx];
 // élément R[i,i] en col‑major = R[row + col*ldr]
    double u = R[i + i * n];
    // élément R[i,i+1]
    double v = R[i + (i+1) * n];
    double t = -rint(v / u);
    U[i + (i+1) * ldu] = t;
    R[i + (i+1) * n] = fma(u, t, v);
}
// Étape 2: Cast U -> Uf
// __global__ void kernelCastUtoFloat(
//     const int64_t* U, double* Uf,
//     int total) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= total) return;
//     Uf[idx] = double(U[idx]);
// }

__global__ void kernelCastUtoFloatAndTranspose(
        const int64_t* __restrict__ U,
        double*        __restrict__ Uf,
        int                     n  // dimension de la matrice (n×n)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    if (idx >= total) return;
    // récupère la ligne et la colonne dans U (row-major)
    int row = idx / n;
    int col = idx % n;
    // écrit dans Uf en column-major => (col, row)
    Uf[col * n + row] = static_cast<double>(U[idx]);
}

// Étape 3a: Gather blocs R11, R12, U22
__global__ void kernelGatherBlocks(
    const double* mat, double* out,
    const int* base_i, const int* base_j,
    int batch, int w, int n)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * w * w;
    if (idx >= total) return;

    // quelle tranche du batch ?
    int bid = idx / (w * w);
    // position i,j dans la matrice w×w
    int rem = idx % (w * w);
    int i   = rem % w;
    int j   = rem / w;

    // copie l’élément (i,j) de la tranche bid
    // offset dans out : bid*w*w + i*w + j
    // offset dans mat : (base_i[bid] + i)*n + (base_j[bid] + j)
    out[idx] = mat[(base_i[bid] + i) + (base_j[bid] + j)*n];
}

__global__ void kernelScatterBlocks(
    const double* in, double* out,
    const int* base_i, const int* base_j,
    int batch, int w, int n)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * w * w;
    if (idx >= total) return;

    int bid = idx / (w * w);
    int rem = idx % (w * w);

    int i = rem % w;
    int j = rem / w;

    int gi = base_i[bid] + i;
    int gj = base_j[bid] + j;

    // on réinjecte in[idx] (i,j) du batch bid au bon endroit
    out[gi + gj * n] = in[idx];
}

// Étape 3d et 5: Round et Cast back traps
__global__ void kernelRound(double* data, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) data[idx] = rint(data[idx]);
}

__global__ void kernelCastFloatToUAndTranspose(
    const double* __restrict__ Uf,  // Uf est en column‑major (n×n)
    int64_t*      __restrict__ U,   // U sera en row‑major (n×n)
    int                   n        // dimension de la matrice
) {
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    if (idx >= total) return;
    // On parcourt Uf en flatten col‑major: idx = row + col*n
    int row = idx % n;
    int col = idx / n;
    // On écrit dans U en row‑major: offset = row*n + col
    U[row * n + col] = static_cast<int64_t>(rint(Uf[idx]));
}

// Étape d'addition pontuelle pour R12 = old S12 + new S12
__global__ void kernelAdd(const double* a, double* b, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) b[idx] += a[idx];
}

//------------------------------------------------------------------------------
// Wrappers pour cublas
//------------------------------------------------------------------------------
void launchBatchedGEMM(
    cublasHandle_t handle,
    const double* A, const double* B,
    double* C, int batch, int w,
    bool strided) {
    const double alpha = 1.0, beta = 0.0;
    if (strided) {
        long long stride = (long long)w * w;
        (cublasDgemmStridedBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            w, w, w,
            &alpha,
            A, w, stride,
            B, w, stride,
            &beta,
            C, w, stride,
            batch));
    } // else you don't have to use it
}

__global__ void fill_ptrs(double **out, double *base, size_t elems, int batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch) {
        out[i] = base + i * elems;
    }
}


void launchBatchedTRSM(
    cublasHandle_t handle,
    double * A, double * B,
    int batch, int w,  double ** dA, double **dB) {
    size_t elems = size_t(w)*w;
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    fill_ptrs<<<blocks, threads>>>(dA, A, elems, batch);
    fill_ptrs<<<blocks, threads>>>(dB, B, elems, batch);
    const double alpha = -1.0;
    (cublasDtrsmBatched(
        handle,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
        w, w,
        &alpha,
        (const double**)dA, w,
        dB, w,
        batch));
    }



//------------------------------------------------------------------------------
// Fonction hôte pour traiter un batch
//------------------------------------------------------------------------------
void processBatch(
    cublasHandle_t handle,
    double* d_R, double* d_Uf,
    const int* bi, const int* bj,
    BatchBuffers &buf,
    int offset, int batch, int w, int n, double** dA, double ** dB) {
    // Gather R11, R12, U22
    int totalElems = batch*w*w;
    const int threads   = 1024;
    int blocks    = (totalElems + threads - 1) / threads;

    kernelGatherBlocks<<<blocks,threads>>>(d_R, buf.d_R11, bi, bi, batch, w, n);
    kernelGatherBlocks<<<blocks,threads>>>(d_R, buf.d_R12, bi, bj, batch, w, n);
    kernelGatherBlocks<<<blocks,threads>>>(d_Uf,buf.d_U22, bj, bj, batch, w, n);
    kernelGatherBlocks<<<blocks,threads>>>(d_Uf,buf.d_U11, bi, bi, batch, w, n);

    // {
    //     char fname[128];
    //     std::snprintf(fname, sizeof(fname), "step0_R11_batch%d_w%d", batch, w);
    //     WRITE_MAT_3D(fname, buf.d_R11, batch, w, w, w);
    // }
    // {
    //     char fname[128];
    //     std::snprintf(fname, sizeof(fname), "step0_R12_batch%d_w%d", batch, w);
    //     WRITE_MAT_3D(fname, buf.d_R12, batch, w, w, w);
    // }
    // {
    //     char fname[128];
    //     std::snprintf(fname, sizeof(fname), "step0_U22_batch%d_w%d", batch, w);
    //     WRITE_MAT_3D(fname, buf.d_U22, batch, w, w, w);
    // }



    // GEMM S12 = R12 * U22
    launchBatchedGEMM(handle, buf.d_R12, buf.d_U22, buf.d_S12, batch, w, true);
    // {
    //     char fname[128];
    //     std::snprintf(fname, sizeof(fname), "step1_S12_batch%d_w%d", batch, w);
    //     WRITE_MAT_3D(fname, buf.d_S12, batch, w, w, w);
    // }
    // Backup old S12
    (cudaMemcpy(buf.d_S12_old, buf.d_S12, size_t(batch)*w*w*sizeof(double), cudaMemcpyDeviceToDevice));

    // Solve triangular U12 = round(-inv(R11)*S12)
    launchBatchedTRSM(handle, buf.d_R11, buf.d_S12, batch, w, dA, dB);
    
    kernelRound<<<blocks,threads>>>(buf.d_S12, totalElems);
    (cudaMemcpy(buf.d_U12, buf.d_S12, size_t(totalElems)*sizeof(double), cudaMemcpyDeviceToDevice));
    // {
    //     char fname[128];
    //     std::snprintf(fname, sizeof(fname), "step2_U12_batch%d_w%d", batch, w);
    //     WRITE_MAT_3D(fname, buf.d_S12, batch, w, w, w);
    // }



    // Compute new R12 = R11 * U12
    launchBatchedGEMM(handle, buf.d_R11, buf.d_U12, buf.d_S12, batch, w, true);
    // Scatter U12 into Uf (R11)
    launchBatchedGEMM(handle, buf.d_U11, buf.d_U12, buf.d_R11, batch, w, true);
    kernelScatterBlocks<<<blocks, threads>>>(buf.d_R11, d_Uf, bi, bj, batch, w, n);
    kernelAdd<<<blocks, threads>>>(buf.d_S12_old, buf.d_S12, totalElems);
    // {
    //     char fname[128];
    //     std::snprintf(fname, sizeof(fname), "step3_R_batch%d_w%d", batch, w);
    //     WRITE_MAT_3D(fname, buf.d_S12, batch, w, w, w);
    // }
    // {
    //     char fname[128];
    //     std::snprintf(fname, sizeof(fname), "step3_U_final_batch%d_w%d", batch, w);
    //     WRITE_MAT(fname, d_Uf, n, n, n);
    // }

    // Scatter result into R
    kernelScatterBlocks<<<blocks, threads>>>(buf.d_S12, d_R, bi, bj, batch, w, n);
    // {
    //     char fname[128];
    //     std::snprintf(fname, sizeof(fname), "step3_R_final_batch%d_w%d", batch, w);
    //     WRITE_MAT(fname, d_R, n, n, n);
    // }
}

//------------------------------------------------------------------------------
// Boucle principale et tail processing
//------------------------------------------------------------------------------
void mainReductionLoop(
    cublasHandle_t handle,
    ReductionPlan &plan,
    double* d_R, int64_t* d_U,
    double* d_Uf,
    const int* d_ranges_i,
    const int* d_ranges_j,
    const int* d_ranges_k) {



    const int threads   = 1024;

    int offset = 0;

    int max_batch = plan.m/2+1;
    int max_w = plan.m / plan.min_batch;
    BatchBuffers* buf = getOrAllocBatchBuffer(plan.m, max_batch, max_w);
    
    double **dA, **dB;
    (cudaMalloc(&dA, max_batch*sizeof(double*)));
    (cudaMalloc(&dB, max_batch*sizeof(double*)));

    for (int level = 1;; ++level) {
        int w = 1 << level;
        int batch = plan.m / w + 1;
        if (batch <= plan.min_batch) break;

        processBatch(handle, d_R, d_Uf,
                     d_ranges_i+offset, d_ranges_j+offset,
                     *buf, offset, batch, w, plan.n, dA, dB);
       

        offset += batch;
    }

    cudaFree(dA);
    cudaFree(dB);
    // WRITE_MAT("Rafterbatch.csv", d_R, plan.n,plan.n,plan.n);
    // WRITE_MAT("Uafterbatch.csv", d_Uf, plan.n,plan.n,plan.n);
    
    int tail = plan.m - offset;
    if (tail > 0) {
        const double alpha = 1.0, beta0 = 0.0, alpha_neg = -1.0;
        std::vector<int> h_tail_i(tail), h_tail_j(tail), h_tail_k(tail);
        (cudaMemcpy(h_tail_i.data(), d_ranges_i + offset, tail * sizeof(int), cudaMemcpyDeviceToHost));
        (cudaMemcpy(h_tail_j.data(), d_ranges_j + offset, tail * sizeof(int), cudaMemcpyDeviceToHost));
        (cudaMemcpy(h_tail_k.data(), d_ranges_k + offset, tail * sizeof(int), cudaMemcpyDeviceToHost));
        for (int t = 0; t < tail; ++t) {
            // 1) On lit les indices i,j,k de la range t
            int i = h_tail_i[t], j = h_tail_j[t], k = h_tail_k[t];
            int h = j - i, w = k - j; //on peut les deviner eux aussi au lieu de revenir au host

            int totalElems = h*w;
            int blocks    = (totalElems + threads - 1) / threads;

            // 2) Préparer les pointeurs sur R et Uf
            double* R11  = d_R  + i     + i     * plan.n;
            // R12 = R[i:i+h, j:j+w]
            double* R12  = d_R  + i     + j     * plan.n;
            // Uf11 = Uf[i:i+h, i:i+h]
            double* Uf11 = d_Uf + i     + i     * plan.n;
            // Uf22 = Uf[j:j+w, j:j+w]
            double* Uf22 = d_Uf + j     + j     * plan.n;

            // 3) Allouer temporaire S12p et R12new sur le GPU
            double *d_S12p, *d_S12old, *d_R12new;
            (cudaMalloc(&d_S12p,    sizeof(double)*h*w));
            (cudaMalloc(&d_S12old,  sizeof(double)*h*w));
            (cudaMalloc(&d_R12new,  sizeof(double)*h*w));

            (cublasDgemm(handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    h, w, w,
                                    &alpha,

                                    R12,   plan.n,
                                    Uf22,  plan.n,
                                    &beta0,
                                    d_S12p, h));

            (cudaMemcpy(d_S12old, d_S12p, sizeof(double)*h*w,
                                cudaMemcpyDeviceToDevice));

            // 6) U12' = round(-inv(R11)*S12')

            (cublasDtrsm(handle,
                                    CUBLAS_SIDE_LEFT,
                                    CUBLAS_FILL_MODE_UPPER,
                                    CUBLAS_OP_N,
                                    CUBLAS_DIAG_NON_UNIT,
                                    h, w,
                                    &alpha_neg,
                                    R11,   plan.n,
                                    d_S12p,   h
                                    ));
            // round in-place
            kernelRound<<<blocks,threads>>>(d_S12p, h*w);


            // 7) R12_new = R11 @ U12'
            (cublasDgemm(handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    h, w, h,
                                    &alpha,
                                    R11,      plan.n,
                                    d_S12p,   h,
                                    &beta0,
                                    d_R12new, h));

            // 8) R[i:j, j:k] = old S12 + new R12
            kernelAdd<<<blocks,threads>>>(d_S12old, d_R12new, h*w);
            (cudaMemcpy2D(
                /*dst=*/ R12,             /* pitch dst = stride en bytes = plan.n * sizeof(double) */
                /*dpitch=*/ plan.n * sizeof(double),
                /*src=*/ d_R12new,        /* pitch src = largeur w * sizeof(double) */
                /*spitch=*/ w * sizeof(double),
                /*width=*/ w * sizeof(double),
                /*height=*/ h,
                cudaMemcpyDeviceToDevice
            ));

            //    on écrit d’abord dans un buffer d_Uf12, puis on la « scatter » en 2D
            double* d_Uf12;
            (cudaMalloc(&d_Uf12, sizeof(double)*h*w));
            (cublasDgemm(handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    h, w, h,
                                    &alpha,
                                    Uf11,     plan.n,
                                    d_S12p,   h,
                                    &beta0,
                                    d_Uf12,   h));
            // copie bloc par cudaMemcpy2D
            (cudaMemcpy2D(
                /*dst*/ d_Uf + i     + j*plan.n,
                /*dpitch*/ sizeof(double)*plan.n,
                /*src*/ d_Uf12,
                /*spitch*/ sizeof(double)*h,
                /*width*/ sizeof(double)*h,
                /*height*/ w,
                cudaMemcpyDeviceToDevice
            ));
            cudaFree(d_S12p);
            cudaFree(d_S12old);
            cudaFree(d_R12new);
            cudaFree(d_Uf12);
        }
    }
    // WRITE_MAT("Uenvraifini.csv", d_Uf, plan.n,plan.n,plan.n);

}

//------------------------------------------------------------------------------
// Fonction principale d'interface
//------------------------------------------------------------------------------
void seysen_reduce_cuda_modular(
    double*    d_R,
    int64_t*   d_U,
    int        n,
    const int* d_ranges_i,
    const int* d_ranges_j,
    const int* d_ranges_k,
    const int* base_cases,
    int        m,
    int        b) {
    const double one = 1.0, zero = 0.0;
    const int threads = 1024;

    cublasHandle_t handle;
    (cublasCreate(&handle));
    double* d_R_cm;
    (cudaMalloc(&d_R_cm, size_t(n)*n*sizeof(double)));
    (cublasDgeam(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,  // op(A)=Aᵀ, pas de B
        n, n,                      // taille de la matrice
        &one,                      // α·Aᵀ
        d_R,   /* lda = */ n,      
        &zero, nullptr, /* ldb = */ n,
        d_R_cm, /* ldc = */ n
    ));

    // cudaFree(d_R);
    d_R = d_R_cm;
    double* d_Uf;
    int total = n * n;
    (cudaMalloc(&d_Uf, size_t(total)*sizeof(double)));

    
    int blk = (total + threads - 1) / threads;
    kernelCastUtoFloatAndTranspose<<<blk,threads>>>(d_U, d_Uf, n);
    if (b > 0) {
        int blocks = (b + threads - 1) / threads;
        kernelProcessBaseCases<<<blocks,threads>>>(d_R, d_Uf, b, base_cases, n, n);
    }
    // // 3) Boucle principale
    ReductionPlan plan{n, m, /*min_batch=*/8};
    mainReductionLoop(handle, plan, d_R, d_U, d_Uf,
                      d_ranges_i, d_ranges_j, d_ranges_k);

    // 5) Uf -> U
    // Uf is good not d_U
    kernelCastFloatToUAndTranspose<<<blk,threads>>>(d_Uf, d_U, n);

    cublasDestroy(handle);
    cudaFree(d_Uf);
    cudaFree(d_R_cm);
}



// Calcule base_cases et triplets (i,j,k) selon __reduction_ranges(n)
static void reduction_ranges(int n,
    std::vector<int>&        base_cases,
    std::vector<std::tuple<int,int,int>>& ranges)
{
    int bit_shift = 1, parts = 1;
    base_cases.clear();
    ranges.clear();

    while (parts < n) {
        int left_bound = 0, left_idx = 0;
        for (int rep = 1; rep <= parts; ++rep) {
            int right_bound = left_bound + 2 * n;
            int mid_idx     = (left_bound + n) >> bit_shift;
            int right_idx   = right_bound >> bit_shift;

            if (right_idx > left_idx + 1) {
                if (right_idx == left_idx + 2) {
                    // intervalle de longueur 2 → base case
                    base_cases.push_back(left_idx);
                } else {
                    // vraie range à réduire
                    ranges.emplace_back(left_idx, mid_idx, right_idx);
                }
            }
            left_bound = right_bound;
            left_idx   = right_idx;
        }
        parts   *= 2;
        bit_shift++;
    }
    // On veut la liste dans l'ordre inverse de l'accumulation
    std::reverse(ranges.begin(), ranges.end());
}


// ----------------------------------------------------------------
// Fonction CPU pour tester si R est Seysen‑réduit :
//    max | R11⁻¹ R12 | <= 1/2
// R stocké en row‑major dans un vecteur de taille n*n
bool is_seysen_reduced_host(const std::vector<double>& R, int n, double tol=0.5) {
    int k = n/2;
    std::vector<double> b(k), x(k);
    double maxabs = 0.0;

    // pour chaque colonne j = k..n-1
    for (int j = k; j < n; ++j) {
        // b = R[0..k-1, j]
        for (int i = 0; i < k; ++i) {
            b[i] = R[i*n + j];
        }
        // résoudre R11 * x = b, où R11 est triangulaire supérieure
        for (int i = k - 1; i >= 0; --i) {
            double s = 0.0;
            for (int l = i+1; l < k; ++l) {
                s += R[i*n + l] * x[l];
            }
            x[i] = (b[i] - s) / R[i*n + i];
            maxabs = std::max(maxabs, std::abs(x[i]));
        }
    }

    std::cout << "max |R11⁻¹ R12| = " << maxabs << "\n";
    return (maxabs < tol);
}
// ----------------------------------------------------------------
// struct RangeBuffers {
//     int* d_ranges_i;
//     int* d_ranges_j;
//     int* d_ranges_k;
//     int* d_base;
//     int m, b;
// };

// // Cache global (par valeur de n), protégé par un mutex pour la sécurité thread
// static std::unordered_map<int, RangeBuffers> g_range_cache;
// static std::mutex                             g_cache_mutex;
struct RangeBuffers {
    int m = 0, b = 0;
    std::vector<int> h_ranges_i, h_ranges_j, h_ranges_k, h_base_cases;
    int *d_ranges_i = nullptr, *d_ranges_j = nullptr, *d_ranges_k = nullptr, *d_base = nullptr;
};
static std::unordered_map<int, RangeBuffers> g_range_cache;
static std::mutex g_range_cache_mutex;

RangeBuffers& getOrAllocRangeBuffers(int n) {
    std::lock_guard<std::mutex> lock(g_range_cache_mutex);
    auto it = g_range_cache.find(n);
    if (it != g_range_cache.end()) return it->second;

    // Compute ONCE for this n
    std::vector<int> base_cases;
    std::vector<std::tuple<int,int,int>> ranges;
    reduction_ranges(n, base_cases, ranges);
    int b = (int)base_cases.size();
    int m = (int)ranges.size();

    RangeBuffers buf;
    buf.m = m; buf.b = b;
    buf.h_ranges_i.resize(m);
    buf.h_ranges_j.resize(m);
    buf.h_ranges_k.resize(m);
    buf.h_base_cases = base_cases;

    for (int t = 0; t < m; ++t)
        std::tie(buf.h_ranges_i[t], buf.h_ranges_j[t], buf.h_ranges_k[t]) = ranges[t];

    // Device malloc + copy (ONCE)
    cudaMalloc(&buf.d_ranges_i, m * sizeof(int));
    cudaMalloc(&buf.d_ranges_j, m * sizeof(int));
    cudaMalloc(&buf.d_ranges_k, m * sizeof(int));
    cudaMalloc(&buf.d_base,     b * sizeof(int));

    cudaMemcpy(buf.d_ranges_i, buf.h_ranges_i.data(), m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(buf.d_ranges_j, buf.h_ranges_j.data(), m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(buf.d_ranges_k, buf.h_ranges_k.data(), m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(buf.d_base,     buf.h_base_cases.data(),  b * sizeof(int), cudaMemcpyHostToDevice);

    // Ajout au cache
    g_range_cache[n] = std::move(buf);
    return g_range_cache[n];
}

void free_all_range_buffers() {
    for (auto& pair : g_range_cache) {
        RangeBuffers& buf = pair.second;
        if (buf.d_ranges_i) cudaFree(buf.d_ranges_i);
        if (buf.d_ranges_j) cudaFree(buf.d_ranges_j);
        if (buf.d_ranges_k) cudaFree(buf.d_ranges_k);
        if (buf.d_base)     cudaFree(buf.d_base);
        buf.d_ranges_i = buf.d_ranges_j = buf.d_ranges_k = buf.d_base = nullptr;
    }
    g_range_cache.clear();
}


extern "C" void seysen_reduce_cuda_cython(
    double*    d_R,
    int64_t*   d_U,
    int        n
) {
    auto& buf = getOrAllocRangeBuffers(n);

    seysen_reduce_cuda_modular(
        d_R, d_U, n,
        buf.d_ranges_i, buf.d_ranges_j, buf.d_ranges_k,
        buf.d_base, buf.m, buf.b
    );
}


void test_seysen_reduce_cuda_cython(double* d_R, int64_t* d_U, int n) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();  // Assure que tout est prêt
    cudaEventRecord(start, 0);

    seysen_reduce_cuda_cython(d_R, d_U, n);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    printf("[CUDA] Temps d'exécution de seysen_reduce_cuda_cython: %.6f ms (%.6f s)\n", elapsed_ms, elapsed_ms/1000.);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(){
    int n = 4096;

    // 1) calcul des ranges et base_cases
    std::vector<int> base_cases;
    std::vector<std::tuple<int,int,int>> ranges;
    reduction_ranges(n, base_cases, ranges);
    int b = base_cases.size();
    int m = ranges.size();

    // 2) génération de R_h aléatoire et U_h identité
    std::srand(42);
    std::vector<double> h_R(n*n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            h_R[i*n + j] = (double(std::rand())/RAND_MAX)*20.0 - 10.0;
        }
    }
    std::vector<int64_t> h_U(n*n, 0);
    for (int i = 0; i < n; ++i) h_U[i*n + i] = 1;
    double*  d_R;
    int64_t* d_U;
    size_t szR = n*n*sizeof(double);
    size_t szU = n*n*sizeof(int64_t);
    (cudaMalloc(&d_R, szR));
    (cudaMalloc(&d_U, szU));

    // Copie hôte → GPU
    (cudaMemcpy(d_R, h_R.data(), szR, cudaMemcpyHostToDevice));
    (cudaMemcpy(d_U, h_U.data(), szU, cudaMemcpyHostToDevice));

    // Appel de la fonction mise en cache
    // WRITE_MAT("step1_R_qr", d_R, n,n,n);
    test_seysen_reduce_cuda_cython(d_R, d_U, n);
    test_seysen_reduce_cuda_cython(d_R, d_U, n);
    cudaDeviceSynchronize();

    // Rapatriement de U



    (cudaMemcpy(h_U.data(), d_U, szU, cudaMemcpyDeviceToHost));
    bool diag_ok = true, differs = false;
    for (int i = 0; i < n; ++i) {
        if (h_U[i*n + i] != 1) diag_ok = false;
        for (int j = 0; j < n; ++j) if (i!=j && h_U[i*n + j] != 0) differs=true;
    }
    printf(diag_ok  ? "OK: diagonale de U = 1" : "Erreur: diagonale de U != 1");
    printf(differs  ? "OK: U off-diagonales ≠ 0" : "U reste identité");

    // write_matrix_csv_c("final_U_CUDA", h_U.data(), n);
    // Test Seysen-réduction de R
    // std::vector<double> h_R_red(n*n);
    // for(int i=0;i<n;++i)for(int j=0;j<n;++j){ double s=0; for(int k=0;k<n;++k) s+=h_R[i*n+k]*h_U[k*n+j]; h_R_red[i*n+j]=s; }
    // h_R.swap(h_R_red);
    // bool sey_ok = is_seysen_reduced_host(h_R, n);
    // printf(sey_ok ? "OK: R est Seysen-réduit" : "Erreur: R n'est pas Seysen-réduit");
    // printf("\n");

    // Libération
    (cudaFree(d_R));
    (cudaFree(d_U));
    free_all_range_buffers();
    free_cached_batch_buffer();

    return 0;
}