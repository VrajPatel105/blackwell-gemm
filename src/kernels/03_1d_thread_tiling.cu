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

    const int  A_view_ty = tx / BLOCK_N;
    const int A_view_tx = tx % BLOCK_N;

    const int B_view_ty =  tx / BLOCK_K;
    const int B_view_tx = tx % BLOCK_K;

    const int row = BLOCK_M * by + COARSE_FACTOR * (tx / BLOCK_M);
    const int col = BLOCK_N * bx + (tx % BLOCK_N);

    __shared__ float shared_mem_A[BLOCK_M][BLOCK_K];
    __shared__ float shared_mem_B[BLOCK_K][BLOCK_N];

    float value[COARSE_FACTOR] = {0.0f};

    for(int ph = 0; ph < ceil(K / BLOCK_K); ph++){

        shared_mem_A[A_view_ty][A_view_tx] = A[by * BLOCK_M + A_view_ty] * K + (ph * BLOCK_K + A_view_tx);

        shared_mem_B[B_view_ty][B_view_tx] = B[ph * BLOCK_K + B_view_ty] * K + (bx * BLOCK_N + B_view_tx);

        __syncthreads();

    }

}

} // namespace

void sgemm_03_1d_thread_tiling(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, cudaStream_t stream){

    const dim3 block(NUM_THREADS);
    const dim3 grid(ceil_div(N, BLOCK_N), ceil_div(M, BLOCK_M));
    gemm_1d_thread_tiling_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK(cudaGetLastError());
}