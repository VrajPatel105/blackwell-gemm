// this is just writing the kernl 03 from scratch withouth any help, just from my understanding. 
// 03_1d_thread_tiling

#include "utils.cuh"
#include "kernels.cuh"

namespace{

int BLOCK_X = 32;
int BLOCK_Y = 32;
int BLOCK_INNER_DIM = 8;
int COARSE_FACTOR = 4;
int NUM_THREADS = BLOCK_X * BLOCK_Y / COARSE_FACTOR;

__global__ void practice_1d_tiling_kernel(size_t M, size_t N, size_t MATRIX_INNER_DIM, float alpha, const float* A, const float* B, float beta, const float* C){

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;    

    int sharedA_row = tx / BLOCK_INNER_DIM;
    int sharedA_col = tx % BLOCK_INNER_DIM;
    
    int sharedB_row = tx / BLOCK_Y;
    int sharedB_col = tx % BLOCK_Y;
    
    int tile_col =

    __shared__ float sharedA[BLOCK_X][BLOCK_INNER_DIM];
    __shared__ float sharedB[BLOCK_INNER_DIM][BLOCK_Y];

    float Pvalue[COARSE_FACTOR] = {0.0f};

    for(int ph = 0; ph < ceil_div(MATRIX_INNER_DIM,  BLOCK_INNER_DIM); ph++){

        // step 1 is to load the shared memory

        sharedA[sharedA_row][sharedA_col] = A[(by * BLOCK_X + sharedA_row) * MATRIX_INNER_DIM + (ph * BLOCK_INNER_DIM + sharedA_col)];
        sharedB[sharedB_row][sharedB_col] = B[(ph * BLOCK_INNER_DIM + sharedB_row) * N + (bx * BLOCK_Y + sharedB_col)];
        __syncthreads();


        for(int k = 0; k < BLOCK_INNER_DIM; k++){
            float B_value_register = sharedB[k][tile_col];
        }

        
    }
    // final c loop
    for()

}
} 

void sgemm_03_1d_thread_tiling(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C, cudaStream_t stream){
    dim3 grid(ceil_div(N, BLOCK_Y), ceil_div(M, BLOCK_X));
    dim3 block(NUM_THREADS);
    practice_1d_tiling_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK(cudaGetLastError());
}