// 03_1d_thread_tiling

#include "utils.cuh"
#include "kernels.cuh"

namespace{

__global__ void gemm_1d_thread_tiling_kernel(size_t M, size_t N, size_t K, float alpha, const float* __restrict__ A, const float* __restrict__ B, float beta, float* __restrict__ C){

}

}

void sgemm_03_1d_thread_tiling(int m, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){


    CUDA_CHECK(cudaGetLastError());
}