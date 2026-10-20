#include "bench.cuh"
#include "utils.cuh"
#include <iostream>
#include <functional>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

using namespace std;

float findMedian(vector<float>ms_per_launch_vec) {

    sort(ms_per_launch_vec.begin(), ms_per_launch_vec.end());

    int n = ms_per_launch_vec.size();

    if (n % 2 != 0) {
        
        return ms_per_launch_vec[n / 2];
    } else {
        
        return (ms_per_launch_vec[(n - 1) / 2] + ms_per_launch_vec[n / 2]) / 2.0;
    }
}

float time_kernels(const function<void()>& launch, const int warmup, const int iterations) {
    
    if (iterations < 1 || warmup < 0) {
        fprintf(stderr, "time_kernels: invalid arguments (warmup=%d, iterations=%d) at %s:%d\n",
                warmup, iterations, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < warmup; i++){
        launch();
    }

    vector<float> ms_per_launch;
    ms_per_launch.reserve(iterations);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for(int i = 0; i < iterations; i++){

    
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaGetLastError());
    
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        
        ms_per_launch.push_back(ms);
        
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float median_ms = findMedian(ms_per_launch);

    return median_ms;
}