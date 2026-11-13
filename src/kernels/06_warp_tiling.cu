// warp tiling

#include "utils.cuh"
#include "kernels.cuh"


namespace{

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 8;

// warp level: which parts of the block tile one warp ownss
constexpr int WARP_SIZE = 32;
constexpr int WARP_M = 32;
constexpr int WARP_N = 64;
constexpr int WM_ITER = 1;
constexpr int WN_ITER = 2;
constexpr int WSUB_M = WARP_M / WM_ITER;
constexpr int WSUB_N = WARP_N / WN_ITER;

// thread level: what one lane computes inside ONE sub tile
constexpr int THREAD_M = 8;
constexpr int THREAD_N = 4;
constexpr int LANES_PER_ROW = WSUB_N / THREAD_N; // 8 lanes side by side in a sub tile

constexpr int WARPS_PER_ROW = BLOCK_N / WARP_N;  // 1
constexpr int NUM_THREADS = (BLOCK_M / WARP_M) * (BLOCK_N / WARP_N) * WARP_SIZE;  // 64

// float4 loading
constexpr int A_FLOAT4_PER_ROW = BLOCK_K / 4; // A is 64 x 8 
constexpr int B_FLOAT4_PER_ROW = BLOCK_N / 4; // B is 8 x 64
constexpr int A_VEC_PASSES = (BLOCK_M * A_FLOAT4_PER_ROW) / NUM_THREADS;
constexpr int B_VEC_PASSES = (BLOCK_K * B_FLOAT4_PER_ROW) / NUM_THREADS;

static_assert((BLOCK_M * A_FLOAT4_PER_ROW) % NUM_THREADS == 0, "A tile must split evenly across threads");
static_assert((BLOCK_K * B_FLOAT4_PER_ROW) % NUM_THREADS == 0, "B tile must split evenly across threads");
static_assert(BLOCK_M % WARP_M == 0 && BLOCK_N % WARP_N == 0, "warp tiles must tile the block tile");
static_assert(WARP_M % WM_ITER == 0 && WARP_N % WN_ITER == 0, "sub tiles must tile the warp tile");
static_assert(WSUB_M % THREAD_M == 0 && WSUB_N % THREAD_N == 0, "thread tiles must tile the sub tile");
static_assert((WSUB_M / THREAD_M) * (WSUB_N / THREAD_N) == WARP_SIZE, "one sub tile must be exactly one warp of lanes");
static_assert(THREAD_M % 4 == 0 && THREAD_N % 4 == 0, "float4 shared reads need multiples of 4");

template <bool VEC>
__global__ void sgemm_06_warp_tiling_kernel(size_t M, size_t N, size_t K, float alpha, const float* __restrict__ A, const float* __restrict__ B, float beta, float* __restrict__ C){

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;

    const int warp_idx = tx / WARP_SIZE;
    const int lane = tx % WARP_SIZE;
    const int warp_row0 = (warp_idx/WARPS_PER_ROW) * WARP_M;
    const int warp_col0 = (warp_idx % WARPS_PER_ROW) * WARP_N;
    const int lane_row = lane / LANES_PER_ROW;
    const int lane_col = lane % LANES_PER_ROW;

    __shared__ __align__(16) float shared_mem_A[BLOCK_K][BLOCK_M]; // this is the transposed view
    __shared__ __align__(16) float shared_mem_B[BLOCK_K][BLOCK_N]; // B shared mem stays the same since it's already in correct format (row major)

    float Pvalue[WM_ITER * THREAD_M][WN_ITER * THREAD_N] = {0.0f};
    float A_registers[WM_ITER][THREAD_M];
    float B_registers[WN_ITER][THREAD_N];

    for(int ph = 0; ph < ceil_div(K, BLOCK_K); ph++){

        if constexpr (VEC){

            // A: one float4 global load -> 4 scalar stores into the transposed tile
            for(int pass = 0; pass < A_VEC_PASSES; pass++){
                
                const int A_view_ty = pass * (NUM_THREADS / A_FLOAT4_PER_ROW) +  (tx / A_FLOAT4_PER_ROW);
                const int A_view_tx = tx % A_FLOAT4_PER_ROW;

                size_t A_global_row = by * BLOCK_M + A_view_ty;
                size_t A_global_col = ph * BLOCK_K + A_view_tx * 4;

                float4 a_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                
                if((A_global_row < M) && (A_global_col < K)){
                    a_vec = *reinterpret_cast<const float4*>(&A[A_global_row * K + A_global_col]);
                }

                shared_mem_A[A_view_tx * 4 + 0][A_view_ty] = a_vec.x;
                shared_mem_A[A_view_tx * 4 + 1][A_view_ty] = a_vec.y;
                shared_mem_A[A_view_tx * 4 + 2][A_view_ty] = a_vec.z;
                shared_mem_A[A_view_tx * 4 + 3][A_view_ty] = a_vec.w;

            }

            for(int pass = 0; pass < B_VEC_PASSES; pass++){

                const int B_view_ty = pass * (NUM_THREADS / B_FLOAT4_PER_ROW) + (tx / B_FLOAT4_PER_ROW);
                const int B_view_tx = tx % B_FLOAT4_PER_ROW;

                size_t B_global_row = ph * BLOCK_K + B_view_ty;
                size_t B_global_col = bx * BLOCK_N + B_view_tx * 4;

                float4 b_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if((B_global_row < K) && (B_global_col < N)){
                    b_vec = *reinterpret_cast<const float4*>(&B[B_global_row * N + B_global_col]);
                }

                *reinterpret_cast<float4*>(&shared_mem_B[B_view_ty][B_view_tx * 4]) = b_vec;
            }

        }
        else{
            
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
                    shared_mem_A[A_view_tx][A_view_ty] = A[A_global_row * K + A_global_col];
                }
                else{
                    shared_mem_A[A_view_tx][A_view_ty] = 0.0f;
                }
                    
                if((B_global_row < K) && (B_global_col < N)){
                    shared_mem_B[B_view_ty][B_view_tx] = B[B_global_row * N + B_global_col];
                }
                else{
                    shared_mem_B[B_view_ty][B_view_tx] = 0.0f;
                }
    
            }

        }

        __syncthreads();

        

}

}


void sgemm_06_warp_tiling(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, cudaStream_t stream){

    const dim3 block(NUM_THREADS);
    const dim3 grid(ceil_div(N, BLOCK_N), ceil_div(M, BLOCK_M));

    if(K % 4 == 0 && N % 4 == 0){
        sgemm_06_warp_tiling_kernel<true><<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    }
    else{
        sgemm_06_warp_tiling_kernel<false><<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    }
    CUDA_CHECK(cudaGetLastError());
}
