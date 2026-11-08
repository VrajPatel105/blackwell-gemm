// 2d thread tiling 

#include "kernels.cuh"
#include "utils.cuh"

namespace{

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 8;
constexpr int COARSE_FACTOR_ROW = 8;
constexpr int COARSE_FACTOR_COL = 8;
constexpr int NUM_THREADS = BLOCK_M * BLOCK_N / (COARSE_FACTOR_ROW * COARSE_FACTOR_COL);

__global__ void gemm_2d_thread_tiling_kernel(size_t M, size_t N, size_t K, float alpha, const float* __restrict__ A, const float* __restrict__ B, float beta, float* __restrict__ C){

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;

    // this is for the pass loop
    const int total_elements_in_shared_mem = BLOCK_M * BLOCK_K;
    const int total_passes = total_elements_in_shared_mem / NUM_THREADS;

    const int row = BLOCK_M * by + COARSE_FACTOR_ROW * (tx / BLOCK_M);
    const int col = BLOCK_N * bx + (tx % BLOCK_N);

    const int tile_col  = tx % BLOCK_N; 
    const int tile_row0 = COARSE_FACTOR_COL * (tx / BLOCK_N);

    __shared__ float shared_mem_A[BLOCK_M][BLOCK_K];
    __shared__ float shared_mem_B[BLOCK_K][BLOCK_N];

    float Pvalue[COARSE_FACTOR_ROW][COARSE_FACTOR_COL] = {0.0f};

    for(int ph = 0; ph < ceil_div(K, BLOCK_K); ph++){

        // here we need the pass loop because now the thread count has reduced so now each thread has to also load multiple elements 
        // compared this to kernel 3, we had one thread load one element so there we never had this problem.
        for(int pass = 0; pass < total_passes; pass++){

            const int A_view_ty = pass * (NUM_THREADS / BLOCK_K) + (tx / BLOCK_K);
            const int A_view_tx = tx % BLOCK_K;

            const int B_view_ty =  pass * (NUM_THREADS / BLOCK_N) + (tx / BLOCK_N);
            const int B_view_tx = tx % BLOCK_N;

            size_t A_global_row = by * BLOCK_M + A_view_ty;
            size_t A_global_col = ph * BLOCK_K + A_view_tx;
            
            size_t B_global_row = ph * BLOCK_K + B_view_ty;
            size_t B_global_col = bx * BLOCK_N + B_view_tx;

            if((A_global_row < M) && (A_global_col < K)){
                shared_mem_A[A_view_ty][A_view_tx] = A[A_global_row * K + A_global_col];
            }
            else{
                shared_mem_A[A_view_ty][A_view_tx] = 0.0f;
            }
                
            if((B_global_row < K) && (B_global_col < N)){
                shared_mem_B[B_view_ty][B_view_tx] = B[B_global_row * N + B_global_col];
            }
            else{
                shared_mem_B[B_view_ty][B_view_tx] = 0.0f;
            }

        }

        __syncthreads();
        
    }

}

}


void sgemm_04_2d_thread_tiling(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, cudaStream_t stream){
    
    const dim3 block(NUM_THREADS);
    const dim3 grid(ceil_div(N, BLOCK_N), ceil_div(M, BLOCK_M));
    gemm_2d_thread_tiling_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK(cudaGetLastError());
}
