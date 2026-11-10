// vectorized shared memory access and float 4

#include "kernels.cuh"
#include "utils.cuh"

namespace{

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 8;
constexpr int COARSE_FACTOR_ROW = 8;
constexpr int COARSE_FACTOR_COL = 8;
constexpr int NUM_THREADS = BLOCK_M * BLOCK_N / (COARSE_FACTOR_ROW * COARSE_FACTOR_COL);
constexpr int PATCHES_PER_ROW = BLOCK_N / COARSE_FACTOR_COL;

// this below vars are on tile level
constexpr int A_FLOAT4_PER_ROW = BLOCK_K / 4; // A is 64 x 8 
constexpr int B_FLOAT4_PER_ROW = BLOCK_N / 4; // B is 8 x 64
constexpr int A_VEC_PASSES = (BLOCK_M * A_FLOAT4_PER_ROW) / NUM_THREADS;
constexpr int B_VEC_PASSES = (BLOCK_K * B_FLOAT4_PER_ROW) / NUM_THREADS;

static_assert((BLOCK_M * A_FLOAT4_PER_ROW) % NUM_THREADS == 0, "A tile must split evenly across threads");
static_assert((BLOCK_K * B_FLOAT4_PER_ROW) % NUM_THREADS == 0, "B tile must split evenly across threads");
static_assert(COARSE_FACTOR_ROW % 4 == 0 && COARSE_FACTOR_COL % 4 == 0, "float4 shared reads need multiples of 4");


// Adding a template since the vectorized load cant work with dimensions like 1000, 777, 513. so by template, we can have a if else version  of a kernel.
// VEC = true : float4 global loads
// VEC = false : implementation stays same as kernel 4 since K % 4 and N % 4 != 0
template <bool VEC>

__global__ void sgemm_05_vectorized_smem_float4_kernel(size_t M, size_t N, size_t  K, float alpha, const float* __restrict__ A, const float* __restrict__ B, float beta, float* __restrict__ C){
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;

    const int total_elements_in_shared_mem = BLOCK_M * BLOCK_K;
    const int total_passes = total_elements_in_shared_mem / NUM_THREADS;

    const int tile_col  = COARSE_FACTOR_COL * (tx % PATCHES_PER_ROW);
    const int tile_row0 = COARSE_FACTOR_ROW * (tx / PATCHES_PER_ROW);

    __shared__ __align__(16) float shared_mem_A[BLOCK_K][BLOCK_M]; // this is the transposed view
    __shared__ __align__(16) float shared_mem_B[BLOCK_K][BLOCK_N]; // B shared mem stays the same since it's already in correct format (row major)

    float Pvalue[COARSE_FACTOR_ROW][COARSE_FACTOR_COL] = {0.0f};

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
            // if the matrix  size does not fit our reqs, the kernel 4 code will run.
            // there is slight change in A's access since we transposed, rest pass loop stays the same as kernel 4
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

        for(size_t k = 0; k < BLOCK_K; k++){
            
            // fill A registers
            float A_registers[COARSE_FACTOR_ROW] = {0.0f}; 
            const float4 a_lo = *reinterpret_cast<const float4*>(&shared_mem_A[k][tile_row0]);
            const float4 a_hi = *reinterpret_cast<const float4*>(&shared_mem_A[k][tile_row0 + 4]);
            A_registers[0] = a_lo.x;
            A_registers[1] = a_lo.y;
            A_registers[2] = a_lo.z;
            A_registers[3] = a_lo.w;

            A_registers[4] = a_hi.x;
            A_registers[5] = a_hi.y;
            A_registers[6] = a_hi.z;
            A_registers[7] = a_hi.w;

            // fill B registers
            float B_registers[COARSE_FACTOR_COL] = {0.0f}; 
            const float4 b_lo = *reinterpret_cast<const float4*>(&shared_mem_B[k][tile_col]);
            const float4 b_hi = *reinterpret_cast<const float4*>(&shared_mem_B[k][tile_col + 4]);

            B_registers[0] = b_lo.x;
            B_registers[1] = b_lo.y;
            B_registers[2] = b_lo.z;
            B_registers[3] = b_lo.w;

            B_registers[4] = b_hi.x;
            B_registers[5] = b_hi.y;
            B_registers[6] = b_hi.z;
            B_registers[7] = b_hi.w;
            // the c loop
            for(size_t i = 0; i < COARSE_FACTOR_ROW; i++){

                float A_register_value = A_registers[i];

                for(size_t j = 0; j < COARSE_FACTOR_COL; j++){

                    Pvalue[i][j] += A_register_value * B_registers[j];

                }

            }


        }
        __syncthreads();

    }

    // final write back loop
    for(int i = 0; i < COARSE_FACTOR_ROW; i++){
        const int global_row =  by * BLOCK_M + tile_row0 + i;

        for(int j = 0; j < COARSE_FACTOR_COL; j++){
            const int global_col = bx * BLOCK_N + tile_col + j;

            if(global_row < M && global_col < N){
                
                float out = alpha * Pvalue[i][j];
                int final_idx = global_row * N + global_col;
                if(beta != 0.0f) out += beta * C[final_idx];
                C[final_idx] = out;
            }

        }

    }
}
}



void sgemm_05_vectorized_smem_float4(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, cudaStream_t stream){

    const dim3 block(NUM_THREADS);
    const dim3 grid(ceil_div(N, BLOCK_N), ceil_div(M, BLOCK_M));

    // float4 needs every row start 16-byte aligned: K % 4 for A, N % 4 for B
    if(K % 4 == 0 && N % 4 == 0){
        sgemm_05_vectorized_smem_float4_kernel<true><<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    }
    else{
        sgemm_05_vectorized_smem_float4_kernel<false><<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    }

    CUDA_CHECK(cudaGetLastError());

}