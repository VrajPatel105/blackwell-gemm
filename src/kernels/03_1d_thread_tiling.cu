// 03_1d_thread_tiling

#include "utils.cuh"
#include "kernels.cuh"

namespace{

constexpr int BLOCK_M = 64; // this is rows of C tile (also A tile)
constexpr int BLOCK_N = 64; // this is cols of C tile (also B tile)
constexpr int BLOCK_K = 8; // this is the depth of each K step
constexpr int COARSE_FACTOR = 8; // this is outputs per thread
constexpr int NUM_THREADS = BLOCK_M * BLOCK_N / COARSE_FACTOR; // number of threads per block = 512

__global__ void gemm_1d_thread_tiling_kernel(size_t M, size_t N, size_t K, float alpha, const float* __restrict__ A, const float* __restrict__ B, float beta, float* __restrict__ C){

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;

    const int  A_view_ty = tx / BLOCK_K; // this is the row the thread is in for A 
    const int A_view_tx = tx % BLOCK_K; // this is the col the thread is in for A

    const int B_view_ty =  tx / BLOCK_N; // row the thread is in for B 
    const int B_view_tx = tx % BLOCK_N; // col the thread is in for B

    const int row = BLOCK_M * by + COARSE_FACTOR * (tx / BLOCK_M);
    const int col = BLOCK_N * bx + (tx % BLOCK_N);

    const int tile_col  = tx % BLOCK_N; // this is used for the main accumulator loop where the thread's column decides which column of B's shared tile. 
    const int tile_row0 = COARSE_FACTOR * (tx / BLOCK_N);// the thread's row decides which row of A's shared tile

    __shared__ float shared_mem_A[BLOCK_M][BLOCK_K];
    __shared__ float shared_mem_B[BLOCK_K][BLOCK_N];

    float Pvalue[COARSE_FACTOR] = {0.0f};
    // Each phase here is representing the computation between the inner dimension (k dim)
    // when we matmul, the inner dimensions of A's and B's tile gets product and those inner dimension values are very big most of the times.
    // for example, if we take 4096 x 512 and 512 x 4096, 512 is the inner dimension, and we cannot load 512 elements all at once in shared memory which is why,
    // we break them down in phases. so here, there will be about 512 / BLOCK_k = 512 / 32 (for ex), so here there are a total of 16 phases. 
    // (we usually keep the block's inner dimension BLOCK_K smalller, therefore )if BLOCK_K is 8, then it will be 512 / 8 = 64 phases 
    for(int ph = 0; ph < ceil_div(K, BLOCK_K); ph++){

        // here, A_global_row is basically the Matrix A's row and same goes for col and B_global_row and col for Matrix B
        size_t A_global_row = by * BLOCK_M + A_view_ty;
        size_t A_global_col = ph * BLOCK_K + A_view_tx;

        // shared_mem_A will load at the A block's row and col which we have calculated earlier above

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

        // final k loop that will calculate the output
        for (size_t k = 0; k < BLOCK_K; k++) {
            float B_value_register = shared_mem_B[k][tile_col];
            for (size_t c = 0; c < COARSE_FACTOR; c++) {
                Pvalue[c] += shared_mem_A[tile_row0 + c][k] * B_value_register;  // A for each of MY 8 rows
            }
        }

        __syncthreads();

    }

    // writing the final finished outputs to c
    for (size_t c = 0; c < COARSE_FACTOR; c++) {
        const int r = row + c; // the row of this output
        if (r < M && col < N) {
            const size_t idx = size_t(r) * N + col;
            float out = alpha * Pvalue[c];
            if (beta != 0.0f) out += beta * C[idx];
            C[idx] = out;
        }
    }

}

} // namespace

void sgemm_03_1d_thread_tiling(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, cudaStream_t stream){

    const dim3 block(NUM_THREADS);
    const dim3 grid(ceil_div(N, BLOCK_N), ceil_div(M, BLOCK_M));
    gemm_1d_thread_tiling_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK(cudaGetLastError());
}

// output
// size              kernel          check  max_abs_err         ms   TFLOP/s  cuBLAS TF/s  % cuBLAS
// 1000x777x513      1d_thread       PASS      3.24e-05     0.1341     5.944        12.08     49.20
// 1024x1024x1024    1d_thread       PASS      8.39e-05     0.3163     6.790        15.91     42.69
// 2048x2048x2048    1d_thread       PASS      1.75e-04     1.9838     8.660        22.54     38.42
// 4096x4096x4096    1d_thread       PASS      0.00e+00    16.1826     8.493        22.94     37.03
// 8192x8192x8192    1d_thread       PASS      0.00e+00   152.0284     7.232        24.60     29.41

// I think the last two results for 4096 and 8192 says that it hit the l2 cache hit rate