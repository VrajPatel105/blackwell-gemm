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