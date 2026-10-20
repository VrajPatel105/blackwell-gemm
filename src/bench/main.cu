#include "bench.cuh"
#include "kernels.cuh"
#include "utils.cuh"
#include <iostream>
#include <cmath>
#include <vector>
using namespace std;


int main() {

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    float alpha = 1;
    float beta = 0;
    int M = 1000;
    int N = 777;
    int K = 513;

    int warmup = 10;
    int iterations = 50;

    // (M x K) (K x N) ==> M x N
    size_t sizeA = size_t(M) * K;
    size_t sizeB = size_t(N) * K;
    size_t sizeC = size_t(M) * N;


    vector<float> h_A(sizeA);
    fill_random(h_A.data(), sizeA, 42);

    vector<float> h_B(sizeB);
    fill_random(h_B.data(), sizeB, 654); 

    vector<float> h_ref(sizeC);
    vector<float> h_out(sizeC);

    float *d_A, *d_B, *d_out;

    CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(float) * sizeB));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * sizeC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(float) * sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeof(float) * sizeB, cudaMemcpyHostToDevice));

    float ms_cublas = time_kernels([&](){
        cublas_sgemm(handle, M, N, K, alpha, d_A, d_B, beta, d_out);},
        warmup, iterations
    );
    // convert ms to second
    float seconds_cublass = ms_cublas / 1000;
    
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_out, sizeof(float) * sizeC, cudaMemcpyDeviceToHost));
    
    cout << "Number" << " | " << "max_abs_err" << " | " << "max_rel_err" << " | " << "Status" <<
         " | " << "raw_ms_custom" << " | " << "raw_ms_cublass" << " | " << " | " << "tflops_custom" << " | " << "tflops_cublass" <<
        " | " << "percentage_of_cublass" << endl;
    for(int i = 0; i < NUM_KERNELS; i++){
        
        CUDA_CHECK(cudaMemset(d_out, 0xFF, sizeof(float) * sizeC)); // poisoning the c_out

        float ms_custom = time_kernels([&]() {
            KERNELS[i].fn(M, N, K, alpha, d_A, d_B, beta, d_out, 0);},
            warmup, iterations
        );
        // convert ms to second
        float seconds_custom = ms_custom / 1000;

        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, sizeof(float) * sizeC, cudaMemcpyDeviceToHost));

        CheckResult results = compare(h_ref.data(), h_out.data(), M, N, K);
        string status = "PASS";
        if(results.pass == 0) status = "FAIL";

        //check and find the TFLOP/s and compare with cublas on howw much %
        double flops = 2 * size_t(M) * N * K;
        double tflops_custom = flops / seconds_custom / 1e12;
        double tflops_cublass = flops / seconds_cublass / 1e12;

        double percentage_of_cublass = tflops_custom / tflops_cublass * 100;

        cout << KERNELS[i].name << " | " << results.max_abs_err << " | " << results.max_rel_err << " | " << status << " | " 
        << ms_custom << " | " << ms_cublas << " | " << tflops_custom << " | " << tflops_cublass << " | " <<  percentage_of_cublass <<  endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);
    cublasDestroy(handle);
    return 0;
}