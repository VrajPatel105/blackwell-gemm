// 02_tiling.cu
// Kernel 02 : Tiling the kernel.

#include "kernels.cuh"
#include "utils.cuh"

namespace{

constexpr int TILE_WIDTH = 16;

__global__ void sgemm_tiling_kernel(size_t M, size_t N, size_t K, float alpha, const float* __restrict__ A, const float* __restrict__ B, float beta, float* __restrict__ C){

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    size_t bx = blockIdx.x; size_t by = blockIdx.y;
    size_t tx = threadIdx.x; size_t ty = threadIdx.y;

    size_t row = by * TILE_WIDTH + ty;
    size_t col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    for (size_t ph = 0; ph < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++ph){

        if((row < M) && (ph * TILE_WIDTH + tx) < K){
            Mds[ty][tx] = A[row * K + (ph * TILE_WIDTH + tx)];
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

    if (row < M && col < N) {
        size_t idx = row * N + col;
        C[idx] = alpha * Pvalue
            + (beta == 0.0f ? 0.0f : beta * C[idx]);
    }
    
    }

} // namespace


void sgemm_02_tiling(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, cudaStream_t stream){

    const dim3 block(TILE_WIDTH, TILE_WIDTH);
    const dim3 grid(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (M + TILE_WIDTH - 1) / TILE_WIDTH
    );
    sgemm_tiling_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK(cudaGetLastError());
}

// output for tile_size = 32
// size              kernel          check  max_abs_err         ms   TFLOP/s  cuBLAS TF/s  % cuBLAS
// 1000x777x513      tiling          PASS      3.24e-05     1.2275     0.649         3.65     17.80
// 1024x1024x1024    tiling          PASS      8.39e-05     1.8316     1.172         4.43     26.48
// 2048x2048x2048    tiling          PASS      1.75e-04    16.1128     1.066         6.00     17.77
// 4096x4096x4096    tiling          PASS      0.00e+00   243.4405     0.565         4.98     11.33
// 8192x8192x8192    tiling          PASS      0.00e+00  2308.1741     0.476         3.38     14.11

// output for tile_size = 16
// size              kernel          check  max_abs_err         ms   TFLOP/s  cuBLAS TF/s  % cuBLAS
// 1000x777x513      tiling          PASS      3.24e-05     0.7898     1.009         3.61     27.94
// 1024x1024x1024    tiling          PASS      8.39e-05     1.7507     1.227         3.64     33.72
// 2048x2048x2048    tiling          PASS      1.75e-04    15.9152     1.079         5.85     18.46
// 4096x4096x4096    tiling          PASS      0.00e+00   339.9859     0.404         5.02      8.05
// 8192x8192x8192    tiling          PASS      0.00e+00  3714.6204     0.296         3.34      8.86


// upon profiling, i found out that tile_size = 16 is better than 32. for smaller size matrices