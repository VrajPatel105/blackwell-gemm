// 01_coalesced.cu
// Kernel 01 : Coalesced kernel


#include "kernels.cuh"
#include "utils.cuh"

namespace{

constexpr int BLOCK = 32;

__global__ void sgemm_coalesced_kernel(size_t M, size_t N, size_t K, float alpha, 
                            const float* __restrict__ A, const float* __restrict__ B, 
                            float beta, float* __restrict__ C){

    // swapped row to col and col to row compared to naive in ordre for the kernel to follow the coasceling pattern
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

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

void sgemm_01_coalesced(int M, int N, int K, float alpha, const float* A, 
            const float* B, float beta, float* C, cudaStream_t stream){
    
    const dim3 block(BLOCK, BLOCK);
    const dim3 grid(ceil_div(N, BLOCK), ceil_div(M, BLOCK)); // this is one of the changes for coasceled compared to naive one. swapping the M and N..
    sgemm_coalesced_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK(cudaGetLastError());

}

// outpout

// size              kernel          check  max_abs_err         ms   TFLOP/s  cuBLAS TF/s  % cuBLAS
// 1000x777x513      coalesced       PASS      3.24e-05     4.2032     0.190         3.67      5.17
// 1024x1024x1024    coalesced       PASS      8.39e-05     2.5677     0.836         4.43     18.86
// 2048x2048x2048    coalesced       PASS      1.75e-04    31.4632     0.546         6.03      9.06
// 4096x4096x4096    coalesced       PASS      0.00e+00   470.2791     0.292         3.85      7.59
// 8192x8192x8192    coalesced       PASS      0.00e+00  11480.3076     0.096         3.21      2.98