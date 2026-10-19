// 00_naive.cu
// Kernel 00 : Naive SGEMM, one thread per output element (single precision gemm)

#include "kernels.cuh"
#include "utils.cuh"

namespace{




}

void sgemm_00_naive(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, cudaStream_t stream){
    
    

}