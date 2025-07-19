#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

#ifndef SEYSEN_KERNEL_H
#define SEYSEN_KERNEL_H

// C wrapper around the C++ implementation
#ifdef __cplusplus
extern "C" {
#endif
void seysen_reduce_cuda_cython(
    double*    d_R,
    int64_t*   d_U,
    int        n
);
#ifdef __cplusplus
}
#endif
#endif  // SEYSEN_KERNEL_H