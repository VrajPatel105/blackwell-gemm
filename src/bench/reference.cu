#include "bench.cuh"
#include "utils.cuh"

// Row-major C = alpha*A*B + beta*C via column-major cuBLAS.
// Row-major C (MxN) has the same memory layout as column-major C^T (NxM),
// and C^T = B^T * A^T. Row-major B and A already look like B^T and A^T
// to cuBLAS, so no transposes are needed at all : just swap the operands.

void cublas_sgemm(cublasHandle_t handler, int M, int N, int K, float alpha,
                  const float* d_A, const float* d_B, float beta, float* d_C) {

  CUBLAS_CHECK(cublasSgemm(handler, CUBLAS_OP_N, CUBLAS_OP_N,
                           N, M, K,        // cuBLAS's m, n, k for C^T (N x M)
                           &alpha,
                           d_B, N,         // B^T: N x K, column length N
                           d_A, K,         // A^T: K x M, column length K
                           &beta,
                           d_C, N));       // C^T: N x M, column length N
}