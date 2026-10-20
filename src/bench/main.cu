// main.cu: benchmark harness.
// For each size: compute a cuBLAS reference, then for each kernel:
// check correctness, and only if it passes, time it against cuBLAS.
// CSV goes to stdout; progress and warnings go to stderr.

#include "bench.cuh"
#include "kernels.cuh"
#include "utils.cuh"
#include <cstdio>
#include <vector>
using namespace std;

struct Size { int M, N, K; };

static const Size SIZES[] = {
    {1000, 777, 513},
    {1024, 1024, 1024},
    {2048, 2048, 2048},
    {4096, 4096, 4096},
};

static int choose_iterations(float one_launch_ms) {
    (void)one_launch_ms;
    return 50;
}

static constexpr int WARMUP = 5;

int main() {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    bool any_fail = false;

    printf("id,name,M,N,K,pass,max_abs_err,max_rel_err,ms,tflops,"
           "cublas_ms,cublas_tflops,pct_cublas\n");

    for (const Size& s : SIZES) {
        const int M = s.M, N = s.N, K = s.K;
        const size_t sizeA = size_t(M) * K;
        const size_t sizeB = size_t(K) * N;
        const size_t sizeC = size_t(M) * N;
        const double flops = 2.0 * M * N * K;

        fprintf(stderr, "== size %dx%dx%d ==\n", M, N, K);

        // host buffers
        vector<float> h_A(sizeA), h_B(sizeB), h_ref(sizeC), h_out(sizeC);
        fill_random(h_A.data(), sizeA, 42);
        fill_random(h_B.data(), sizeB, 654);

        // device buffers
        float *d_A, *d_B, *d_ref, *d_out;
        CUDA_CHECK(cudaMalloc(&d_A,   sizeof(float) * sizeA));
        CUDA_CHECK(cudaMalloc(&d_B,   sizeof(float) * sizeB));
        CUDA_CHECK(cudaMalloc(&d_ref, sizeof(float) * sizeC));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * sizeC));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(float) * sizeA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeof(float) * sizeB, cudaMemcpyHostToDevice));

        // reference run from cublass
        auto run_cublas = [&]() {
            cublas_sgemm(handle, M, N, K, alpha, d_A, d_B, beta, d_ref);
        };
        run_cublas();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, sizeof(float) * sizeC, cudaMemcpyDeviceToHost));

        // timing cublas stats
        const float cublas_est_ms = time_kernels(run_cublas, 1, 1);
        const float cublas_ms = time_kernels(run_cublas, WARMUP, choose_iterations(cublas_est_ms));
        const double cublas_tflops = flops / (cublas_ms / 1e3) / 1e12;

        // running our kernels
        for (int i = 0; i < NUM_KERNELS; ++i) {
            const KernelEntry& e = KERNELS[i];
            fprintf(stderr, "  %-14s ... ", e.name);

            auto run_kernel = [&]() {
                e.fn(M, N, K, alpha, d_A, d_B, beta, d_out, 0);
            };

            // poisoning the d_out
            CUDA_CHECK(cudaMemset(d_out, 0xFF, sizeof(float) * sizeC));
            run_kernel();
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, sizeof(float) * sizeC, cudaMemcpyDeviceToHost));
            const CheckResult r = compare(h_ref.data(), h_out.data(), M, N, K);

            if (!r.pass) {
                any_fail = true;
                fprintf(stderr, "FAIL (max_abs_err %.3e)\n", r.max_abs_err);
                    
                printf("%d,%s,%d,%d,%d,0,%.6e,%.6e,,,%.6f,%.4f,\n",
                       e.id, e.name, M, N, K, r.max_abs_err, r.max_rel_err,
                       cublas_ms, cublas_tflops);
                continue;
            }

            // timing the passed kernels
            const float est_ms = time_kernels(run_kernel, 1, 1);
            const float ms = time_kernels(run_kernel, WARMUP, choose_iterations(est_ms));
            const double tflops = flops / (ms / 1e3) / 1e12;
            const double pct = tflops / cublas_tflops * 100.0;

            fprintf(stderr, "PASS  %.4f ms  %.3f TFLOP/s  %.2f%% of cuBLAS\n", ms, tflops, pct);
            printf("%d,%s,%d,%d,%d,1,%.6e,%.6e,%.6f,%.4f,%.6f,%.4f,%.2f\n",
                   e.id, e.name, M, N, K, r.max_abs_err, r.max_rel_err,
                   ms, tflops, cublas_ms, cublas_tflops, pct);
        }

        // free up the bufferes
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_ref));
        CUDA_CHECK(cudaFree(d_out));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return any_fail ? 1 : 0;
}