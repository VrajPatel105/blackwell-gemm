// 02_tiling.cu
// Kernel 02 : Tiling the kernel.

#include "kernels.cuh"
#include "utils.cuh"

namespace{

constexpr int TILE_WIDTH = 4;

__global__ void sgemm_tiling_kernel(size_t M, size_t N, size_t K, float alpha, const float* __restrict__ A, const float* __restrict__ B, float beta, float* __restrict__ C){

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    size_t bx = blockIdx.x; size_t by = blockIdx.y;
    size_t tx = threadIdx.x; size_t ty = threadIdx.y;

    size_t row = by * TILE_WIDTH + ty;
    size_t col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    for(size_t ph = 0; ph < K / TILE_WIDTH; ++ph){

        if((row < M) && (ph * TILE_WIDTH + tx) < K){
            Mds[ty][tx] = A[row * M + ph * TILE_WIDTH + tx];
        }
        else{
            Mds[ty][tx] = 0.0f;
        }
        if((ph * TILE_WIDTH + ty) < K && (col < N)){
            Nds[ty][tx] = B[(ph * TILE_WIDTH + ty) * N + col];
        }
        else{
            Nds[ty][tx] = 0.0f;
        }
        __syncthreads();

        for(size_t k = 0; k < TILE_WIDTH; ++k){
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();

    }
    if((row < M) && (col < N)){
        C[row * K + col] = Pvalue;
    }
    
    }

} // namespace


void sgemm_02_tiling(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, cudaStream_t stream){

    const dim3 block(TILE_WIDTH, TILE_WIDTH);
    const dim3 grid((N / TILE_WIDTH), (M / TILE_WIDTH));
    sgemm_tiling_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK(cudaGetLastError());
}