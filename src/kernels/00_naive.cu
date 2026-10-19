// 00_naive.cu
// Kernel 00 : Naive SGEMM, one thread per output element (single precision gemm)

#include "kernels.cuh"
#include "utils.cuh"

namespace{

constexpr int BLOCK = 32;

__global__ void sgemm_naive_kernel(size_t M, size_t N, size_t K, float alpha, 
                            const float* __restrict__ A, const float* __restrict__ B, 
                            float beta, float* __restrict__ C){

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row >= M || col >= N) return;

    float accu = 0.0f;
    for(int k = 0; k < K; ++k){
        accu += A[ row * M + k] * B[ k * N + col];
    }
    
    float out = alpha * accu;
    if(beta != 0.0f) out += beta * C[row * N * col];
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