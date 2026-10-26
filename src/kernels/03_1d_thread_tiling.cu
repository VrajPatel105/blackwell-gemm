// 03_1d_thread_tiling

#include "utils.cuh"
#include "kernels.cuh"

namespace{

constexpr int BM = 64; // this is rows of C tile (also A tile)
constexpr int BN = 64; // this is cols of C tile (also B tile)
constexpr int BK = 8; // this is the depth of each K step
constexpr int TM = 8; // this is outputs per thread
constexpr int NUM_THREADS = BM * BN / TM; // number of threads per block = 512

__global__ void gemm_1d_thread_tiling_kernel(size_t M, size_t N, size_t K, float alpha, const float* __restrict__ A, const float* __restrict__ B, float beta, float* __restrict__ C){

}

} // namespace

void sgemm_03_1d_thread_tiling(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, cudaStream_t stream){

    const dim3 block(NUM_THREADS);
    const dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    gemm_1d_thread_tiling_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK(cudaGetLastError());
}