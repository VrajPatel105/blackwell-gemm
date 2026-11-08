// 2d thread tiling 

#include "kernels.cuh"
#include "utils.cuh"

namespace{

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 8;
constexpr int COARSE_FACTOR = 8;
constexpr int NUM_THREADS = BLOCK_M * BLOCK_N / COARSE_FACTOR;

__global__ void gemm_2d_thread_tiling_kernel(size_t M, size_t N, size_t K, float alpha, const float* __restrict__ A, const float* __restrict__ B, float beta, float* __restrict__ C){

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;

    const int  A_view_ty = tx / BLOCK_K;
    const int A_view_tx = tx % BLOCK_K;

    const int B_view_ty =  tx / BLOCK_N;
    const int B_view_tx = tx % BLOCK_N;

    const int row = BLOCK_M * by + COARSE_FACTOR * (tx / BLOCK_M);
    const int col = BLOCK_N * bx + (tx % BLOCK_N);

    const int tile_col  = tx % BLOCK_N; 
    const int tile_row0 = COARSE_FACTOR * (tx / BLOCK_N);

    __shared__ float shared_mem_A[BLOCK_M][BLOCK_K];
    __shared__ float shared_mem_B[BLOCK_K][BLOCK_N];

    float Pvalue[COARSE_FACTOR] = {0.0f};

    for(int ph = 0; ph < ceil_div(K, BLOCK_K); ph++){

        size_t A_global_row = by * BLOCK_M + A_view_ty;
        size_t A_global_col = ph * BLOCK_K + A_view_tx;

        if((A_global_row < M) && (A_global_col < K)){
            shared_mem_A[A_view_ty][A_view_tx] = A[A_global_row * K + A_global_col];
        }
        else{
            shared_mem_A[A_view_ty][A_view_tx] = 0.0f;
        }

        size_t B_global_row = ph * BLOCK_K + B_view_ty;
        size_t B_global_col = bx * BLOCK_N + B_view_tx;

        if((B_global_row < K) && (B_global_col < N)){
            shared_mem_B[B_view_ty][B_view_tx] = B[B_global_row * N + B_global_col];
        }
        else{
            shared_mem_B[B_view_ty][B_view_tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_K; k++) {
            float B_value_register = shared_mem_B[k][tile_col];
            for (int c = 0; c < COARSE_FACTOR; c++) {
                Pvalue[c] += shared_mem_A[tile_row0 + c][k] * B_value_register;
            }
        }

        __syncthreads();

    }

    for (int c = 0; c < COARSE_FACTOR; c++) {
        const int r = row + c;
        if (r < M && col < N) {
            const size_t idx = size_t(r) * N + col;
            float out = alpha * Pvalue[c];
            if (beta != 0.0f) out += beta * C[idx];
            C[idx] = out;
        }
    }

}

}


void sgemm_04_2d_thread_tiling(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, cudaStream_t stream){
    
    const dim3 block(NUM_THREADS);
    const dim3 grid(ceil_div(N, BLOCK_N), ceil_div(M, BLOCK_M));
    gemm_2d_thread_tiling_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK(cudaGetLastError());
}
