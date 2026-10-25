// 00_naive.cu
// Kernel 00 : Naive SGEMM, one thread per output element (single precision gemm)

#include "kernels.cuh"
#include "utils.cuh"

namespace{

constexpr int BLOCK = 32;

__global__ void sgemm_naive_kernel(size_t M, size_t N, size_t K, float alpha, 
                            const float* __restrict__ A, const float* __restrict__ B, 
                            float beta, float* __restrict__ C){

    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row >= M || col >= N) return;

    float accu = 0.0f;
    for(size_t k = 0; k < K; ++k){
        accu += A[ row * K + k] * B[ k * N + col];
    }
    
    float out = alpha * accu;
    if(beta != 0.0f) out += beta * C[row * N + col];
    C[row * N + col] = out;

}

} // namespace

void sgemm_00_naive(int M, int N, int K, float alpha, const float* A, 
            const float* B, float beta, float* C, cudaStream_t stream){
    
    const dim3 block(BLOCK, BLOCK);
    const dim3 grid(ceil_div(M, BLOCK), ceil_div(N, BLOCK));
    sgemm_naive_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK(cudaGetLastError());

}

// output 

// size              kernel          check  max_abs_err         ms   TFLOP/s  cuBLAS TF/s  % cuBLAS
// 1000x777x513      naive           PASS      3.24e-05     5.8065     0.137         3.05      4.50
// 1024x1024x1024    naive           PASS      8.39e-05    34.4608     0.062         5.68      1.10
// 2048x2048x2048    naive           PASS      1.75e-04   383.9169     0.045         4.10      1.09
// 4096x4096x4096    naive           PASS      0.00e+00   869.4537     0.158         5.04      3.14
// 8192x8192x8192    naive           PASS      0.00e+00  7485.0913     0.147         1.64      8.98