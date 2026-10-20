#include "bench.cuh"
#include "kernels.cuh"
#include "utils.cuh"
#include <iostream>
#include <cmath>
#include <vector>
using namespace std;

int main() {

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);

    float alpha = 1;
    float beta = 0;
    int M = 1000;
    int N = 777;
    int K = 513;

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
    
    if(cudaMalloc(&d_A, sizeof(float) * sizeA) != cudaSuccess){
        cout << "Cuda memory for d_A not allocated" << endl;
        return 1;
    }
    if(cudaMalloc(&d_B, sizeof(float) * sizeB) != cudaSuccess){
        cout << "Cuda memory for d_B not allocated" << endl;
        return 1;
    }
    if(cudaMalloc(&d_out, sizeof(float) * sizeC) != cudaSuccess){
        cout << "Cuda memory for d_C not allocated" << endl;
        return 1;
    }

    if(cudaMemcpy(d_A, h_A.data(), sizeof(float) * sizeA, cudaMemcpyHostToDevice) != cudaSuccess){
        cout << "Unable to copy memory from h_A to d_A " << endl;
        return 1;
    }
    if(cudaMemcpy(d_B, h_B.data(), sizeof(float) * sizeB, cudaMemcpyHostToDevice) != cudaSuccess){
        cout << "Unable to copy memory from h_B to d_B " << endl;
        return 1;
    }

    
    cublas_sgemm(handle, M, N, K, alpha, d_A, d_B, beta, d_out);

    if(cudaMemcpy(h_ref.data(), d_out, sizeof(float) * sizeC, cudaMemcpyDeviceToHost) != cudaSuccess){
        cout << "Unable to copy memory from d_out to h_ref " << endl;
        return 0;
    }

    cout << "Number" << " | " << "max_abs_err" << " | " << "max_rel_err" << " | " << "Status";
    for(int i = 0; i < NUM_KERNELS; i++){
        
        CUDA_CHECK(cudaMemset(d_out, 0xFF, sizeof(float) * sizeC));
        
        KERNELS[i].fn(M, N, K, alpha, d_A, d_B, beta, d_out, stream);
        if(cudaMemcpy(h_out.data(), d_out, sizeof(float) * sizeC, cudaMemcpyDeviceToHost) != cudaSuccess){
            cout << "Unable to copy memory from d_out to h_out " << endl;
            return 0;
        }

        CheckResult results = compare(h_ref.data(), h_out.data(), M, N, K);
        string status = "PASS";
        if(results.pass == 0) status = "FAIL";
        cout << i << " | " << results.max_abs_err << " | " << results.max_rel_err << " | " << status;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    return 0;
}