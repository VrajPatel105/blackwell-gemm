// vectorized shared memory access and float 4

#include "kernels.cuh"
#include "utils.cuh"


__global__ void sgemm_05_vectorized_smem_float4_kernel(size_t M, size_t N, size_t  K, float alpha, const float* __restrict__ A, const float* __restrict__ B, float beta, float* __restrict__ C){
    
}


void sgemm_05_vectorized_smem_float4(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, cudaStream_t stream){

    CUDA_CHECK(cudaGetLastError());

}