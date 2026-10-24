// 02_tiling.cu
// Kernel 02 : Tiling the kernel.

#include "kernels.cuh"
#include "utils.cuh"

namespace{

constexpr int BLOCK = 32;
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

        Mds[tx][ty] = A[row * M + ph * TILE_WIDTH + tx];
        Nds[tx][ty] = B[(ph * TILE_WIDTH + ty) * N + col];
        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; ++k){
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();

    }

    C[row * K + col] = Pvalue;

    }

} // namespace


void sgemm_02_tiling(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, cudaStream_t stream){


    CUDA_CHECK(cudaGetLastError());
}