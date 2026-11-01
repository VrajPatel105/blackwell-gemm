// this is just writing the kernl 03 from scratch withouth any help, just from my understanding. 
// 03_1d_thread_tiling

#include "utils.cuh"
#include "kernels.cuh"

namespace{
    __global__ void practice_1d_tiling_kernel(size_t M, size_t N, size_t K, float alpha, const float* A, const float* B, float beta, const float* C){

    }
} 

void sgemm_03_1d_thread_tiling(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, cudaStream_t stream){
    dim3 grid();
    dim3 block();
    practice_1d_tiling_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK(cudaGetLastError());
}