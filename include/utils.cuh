#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err_ = (call);                                             \
    if (err_ != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA error: %s at %s:%d\n",                         \
              cudaGetErrorString(err_), __FILE__, __LINE__);               \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

#define CUBLAS_CHECK(call)                                                 \
  do {                                                                     \
    cublasStatus_t st_ = (call);                                           \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                    \
      fprintf(stderr, "cuBLAS error: %s at %s:%d\n",                       \
              cublasGetStatusString(st_), __FILE__, __LINE__);             \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

__host__ __device__ constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }