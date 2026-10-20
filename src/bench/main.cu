#include "bench.cuh"
#include <iostream>
#include <cmath>
#include <vector>
using namespace std;

int main() {

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1;
    float beta = 0;
    int d_M = 1000;
    int d_N = 777;
    int d_K = 513;

    size_t sizeA = size_t(d_M) * d_K;
    vector<float> h_A(sizeA);
    fill_random(h_A.data(), sizeA, 42);   

    size_t sizeB = size_t(d_N) * d_K;
    vector<float> h_B(sizeB);
    fill_random(h_B.data(), sizeB, 432); 

    size_t sizeC = size_t(d_M) * d_N;
    vector<float> h_C(sizeC);


    float* h_A = h_A.data(); // this gets us the pointer
    float* h_B = h_B.data(); // this gets us the pointer

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, sizeof(float) * sizeA);
    cudaMalloc(&d_B, sizeof(float) * sizeB);
    cudaMalloc(&d_C, sizeof(float) * sizeC);

    cudaMemcpy(d_A, h_A, sizeof(float) * sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * sizeB, cudaMemcpyHostToDevice);

    cublas_sgemm(handle, d_M, d_N, d_K, alpha, d_A, d_B, beta, d_C);

    cudaMemcpy(h_C, d_C, sizeof(float) * sizeC, cudaMemcpyDeviceToHost);

    delete[] d_A;
    delete[] d_B;
    delete[] d_C;
    cublasDestroy(handle);
    return 0;
}