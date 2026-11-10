// warp tiling

#include "utils.cuh"
#include "kernels.cuh"


namespace{

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 8;

__global__ void sgemm_06_warp_tiling_kernel(size_t M, size_t N, size_t K, float alpha, const float* __restrict__ A, const float* __restrict__ B, float beta, float* __restrict__ C){

}

}


void sgemm_06_warp_tiling(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, cudaStream_t stream){

    const dim3 block(NUM_THREADS);
    const dim3 grid(ceil_div(N, BLOCK_N), ceil_div(M, BLOCK_M));

    if(K % 4 == 0 && N % 4 == 0){
        sgemm_06_warp_tiling_kernel<true><<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    }
    else{
        sgemm_06_warp_tiling_kernel<false><<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    }
    CUDA_CHECK(cudaGetLastError());
}
