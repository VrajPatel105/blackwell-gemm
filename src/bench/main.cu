// main.cu: benchmark harness.
// For each size: compute a cuBLAS reference, then for each kernel:
// check correctness, and only if it passes, time it against cuBLAS.
// CSV goes to stdout; progress and warnings go to stderr.

#include "bench.cuh"
#include "kernels.cuh"
#include "utils.cuh"
#include <cstdio>
#include <vector>
#include <cstring>  
#include <cstdlib>  
using namespace std;

struct Size { int M, N, K; };

static const Size SIZES[] = {
    {1000, 777, 513},
    {1024, 1024, 1024},
    {2048, 2048, 2048},
    {4096, 4096, 4096},
    {8192, 8192, 8192},
};

static int choose_iterations(float one_launch_ms) {
    constexpr float TARGET_MS = 200.0f;
    constexpr int   MIN_ITERS = 5;
    constexpr int   MAX_ITERS = 10000;

    if (one_launch_ms <= 0.0f) return MAX_ITERS;
    int iters = int(TARGET_MS / one_launch_ms);
    if (iters < MIN_ITERS) iters = MIN_ITERS;
    if (iters > MAX_ITERS) iters = MAX_ITERS;
    return iters;
}

static constexpr int WARMUP = 5;

struct Args {
    int kernel_id = -1;
    bool profile = false;
    int size = 0;
    bool csv = false;
};

static void usage(const char* prog) {
    fprintf(stderr,
        "usage: %s [kernel_id|all] [--size S] [--profile S]\n"
        "  %s            run all kernels at the default sizes\n"
        "  %s 0          run kernel 0 at the default sizes\n"
        "  %s 0 --size 2048\n"
        "  %s 0 --profile 4096   one launch only, for ncu\n",
        prog, prog, prog, prog, prog);
}

static void print_header(bool csv) {
    if (csv) {
        printf("id,name,M,N,K,pass,max_abs_err,max_rel_err,ms,tflops,"
               "cublas_ms,cublas_tflops,pct_cublas\n");
    } else {
        printf("%-16s  %-14s  %-5s  %11s  %9s  %8s  %11s  %8s\n",
               "size", "kernel", "check", "max_abs_err", "ms",
               "TFLOP/s", "cuBLAS TF/s", "% cuBLAS");
    }
}

// pass == false prints blanks for the timing columns
static void print_row(bool csv, const KernelEntry& e, int M, int N, int K,
                      const CheckResult& r, float ms, double tflops,
                      float cublas_ms, double cublas_tflops, double pct) {
    if (csv) {
        if (r.pass)
            printf("%d,%s,%d,%d,%d,1,%.6e,%.6e,%.6f,%.4f,%.6f,%.4f,%.2f\n",
                   e.id, e.name, M, N, K, r.max_abs_err, r.max_rel_err,
                   ms, tflops, cublas_ms, cublas_tflops, pct);
        else
            printf("%d,%s,%d,%d,%d,0,%.6e,%.6e,,,%.6f,%.4f,\n",
                   e.id, e.name, M, N, K, r.max_abs_err, r.max_rel_err,
                   cublas_ms, cublas_tflops);
        return;
    }
    char size[32];
    snprintf(size, sizeof(size), "%dx%dx%d", M, N, K);
    if (r.pass)
        printf("%-16s  %-14s  %-5s  %11.2e  %9.4f  %8.3f  %11.2f  %8.2f\n",
               size, e.name, "PASS", r.max_abs_err, ms, tflops, cublas_tflops, pct);
    else
        printf("%-16s  %-14s  %-5s  %11.2e  %9s  %8s  %11.2f  %8s\n",
               size, e.name, "FAIL", r.max_abs_err, "-", "-", cublas_tflops, "-");
    fflush(stdout);   // show each row as soon as it's ready
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const char* s = argv[i];
        if (strcmp(s, "--profile") == 0 && i + 1 < argc) {
            a.profile = true;
            a.size = atoi(argv[++i]);
        } else if (strcmp(s, "--size") == 0 && i + 1 < argc) {
            a.size = atoi(argv[++i]);
        } else if (strcmp(s, "--csv") == 0) {
            a.csv = true;
        } else if (strcmp(s, "all") == 0) {
            a.kernel_id = -1;
        } else if (s[0] >= '0' && s[0] <= '9') {
            a.kernel_id = atoi(s);
        } else {
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    if (a.profile && a.kernel_id < 0) {
        fprintf(stderr, "--profile needs a specific kernel id\n");
        exit(EXIT_FAILURE);
    }
    if ((a.profile || a.size != 0) && a.size <= 0) {
        fprintf(stderr, "size must be a positive integer\n");
        exit(EXIT_FAILURE);
    }
    return a;
}


int main(int argc, char** argv) {

    const Args args = parse_args(argc, argv);
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    bool any_fail = false;

    // profile mode: exactly one launch of one kernel, nothing else (for ncu)
    if (args.profile) {
        const KernelEntry* entry = nullptr;
        for (int i = 0; i < NUM_KERNELS; ++i)
            if (KERNELS[i].id == args.kernel_id) entry = &KERNELS[i];
        if (!entry) {
            fprintf(stderr, "no kernel with id %d\n", args.kernel_id);
            CUBLAS_CHECK(cublasDestroy(handle));
            return 1;
        }

        const int M = args.size, N = args.size, K = args.size;
        const size_t sizeA = size_t(M) * K;
        const size_t sizeB = size_t(K) * N;
        const size_t sizeC = size_t(M) * N;

        vector<float> h_A(sizeA), h_B(sizeB);
        fill_random(h_A.data(), sizeA, 42);
        fill_random(h_B.data(), sizeB, 654);

        float *d_A, *d_B, *d_out;
        CUDA_CHECK(cudaMalloc(&d_A,   sizeof(float) * sizeA));
        CUDA_CHECK(cudaMalloc(&d_B,   sizeof(float) * sizeB));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * sizeC));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(float) * sizeA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeof(float) * sizeB, cudaMemcpyHostToDevice));

        fprintf(stderr, "profiling %s at %d^3 (single launch)\n", entry->name, args.size);
        entry->fn(M, N, K, alpha, d_A, d_B, beta, d_out, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_out));
        CUBLAS_CHECK(cublasDestroy(handle));
        return 0;
    }

    vector<Size> sizes;
    if (args.size > 0) sizes.push_back({args.size, args.size, args.size});
    else               sizes.assign(begin(SIZES), end(SIZES));

    print_header(args.csv);

    for (const Size& s : sizes) {
        const int M = s.M, N = s.N, K = s.K;
        const size_t sizeA = size_t(M) * K;
        const size_t sizeB = size_t(K) * N;
        const size_t sizeC = size_t(M) * N;
        const double flops = 2.0 * M * N * K;

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
            if (args.kernel_id >= 0 && e.id != args.kernel_id) continue;


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
                print_row(args.csv, e, M, N, K, r, 0, 0, cublas_ms, cublas_tflops, 0);
                continue;
            }

            // timing the passed kernels
            const float est_ms = time_kernels(run_kernel, 1, 1);
            const float ms = time_kernels(run_kernel, WARMUP, choose_iterations(est_ms));
            const double tflops = flops / (ms / 1e3) / 1e12;
            const double pct = tflops / cublas_tflops * 100.0;

            print_row(args.csv, e, M, N, K, r, ms, tflops, cublas_ms, cublas_tflops, pct);
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