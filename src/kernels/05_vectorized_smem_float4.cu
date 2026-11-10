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
constexpr int B_VEC_PASSES = (BLOCK_N * B_FLOAT4_PER_ROW) / NUM_THREADS;
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
                const int A_view_tx = tx / A_FLOAT4_PER_ROW;

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
            for(size_t a = 0; a < COARSE_FACTOR_ROW; a++){
                A_registers[a] = shared_mem_A[tile_row0 + a][k];
            }

            // fill B registers
            float B_registers[COARSE_FACTOR_COL] = {0.0f}; 
            for(size_t b = 0; b < COARSE_FACTOR_COL; b++){
                B_registers[b] = shared_mem_B[k][tile_col + b];
            }

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
    sgemm_05_vectorized_smem_float4_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK(cudaGetLastError());

}