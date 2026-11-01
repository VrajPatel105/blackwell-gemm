// 2d thread tiling 

#include "kernels.cuh"
#include "utils.cuh"

namespace{



}


void sgemm_04_2d_thread_tiling(int m, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){

    
    CUDA_CHECK(cudaGetLastError());
}
