#include "bench.cuh"
#include "utils.cuh"
#include <cstdio>

int main() {
    const size_t bytes = 1ULL << 30;
    void *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));
    CUDA_CHECK(cudaMemset(d_src, 0, bytes));

    const float copy_ms = time_kernels([&]() {
        CUDA_CHECK(cudaMemcpyAsync(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice, 0));
    }, 3, 20);
    const double bandwidth = 2.0 * bytes / (copy_ms / 1e3) / 1e9;

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));

    const int N = 8192;
    const size_t elems = size_t(N) * N;
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_A, 0, elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_B, 0, elems * sizeof(float)));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    const float gemm_ms = time_kernels([&]() {
        cublas_sgemm(handle, N, N, N, 1.0f, d_A, d_B, 0.0f, d_C);
    }, 3, 10);
    const double tflops = 2.0 * N * N * N / (gemm_ms / 1e3) / 1e12;

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    printf("bandwidth: %.1f GB/s  (device-to-device copy, 1 GiB, %.3f ms)\n", bandwidth, copy_ms);
    printf("fp32 peak: %.2f TFLOP/s  (cuBLAS SGEMM %d^3, %.3f ms)\n", tflops, N, gemm_ms);
    return 0;
}

// output on running it for about 3 times.
// bandwidth: 779.2 GB/s  (device-to-device copy, 1 GiB, 2.756 ms)
// fp32 peak: 28.36 TFLOP/s  (cuBLAS SGEMM 8192^3, 38.769 ms)
// (gemm-env) vraj@Vraj:/mnt/c/dev/projects/blackwell-gemm$ cmake --build build --target ceilings -j && ./build/ceilings
// [0/2] Re-checking globbed directories...
// ninja: no work to do.
// bandwidth: 779.2 GB/s  (device-to-device copy, 1 GiB, 2.756 ms)
// fp32 peak: 28.36 TFLOP/s  (cuBLAS SGEMM 8192^3, 38.764 ms)
// (gemm-env) vraj@Vraj:/mnt/c/dev/projects/blackwell-gemm$ cmake --build build --target ceilings -j && ./build/ceilings
// [0/2] Re-checking globbed directories...
// ninja: no work to do.
// bandwidth: 778.9 GB/s  (device-to-device copy, 1 GiB, 2.757 ms)
// fp32 peak: 28.36 TFLOP/s  (cuBLAS SGEMM 8192^3, 38.768 ms)
// (gemm-env) vraj@Vraj:/mnt/c/dev/projects/blackwell-gemm$ 

// therefore: 
// 28.4 TFLOP/s ÷ 779 GB/s ≈ 36 FLOPs per byte.