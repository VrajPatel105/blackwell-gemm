#pragma once
#include <cuda_runtime.h>

using GemmFn = void (*)(int M, int N, int K, float alpha, const float* A,
                        const float* B, float beta, float* C, cudaStream_t s);

void sgemm_00_naive (int, int, int, float, const float*, const float*, float, float*, cudaStream_t);
void sgemm_01_coalesced (int, int, int, float, const float*, const float*, float, float*, cudaStream_t);


struct KernelEntry { int id; const char* name; GemmFn fn; };

inline const KernelEntry KERNELS[] = {
  {0, "naive", sgemm_00_naive},
  {1, "coalesced", sgemm_01_coalesced}
};
inline constexpr int NUM_KERNELS = sizeof(KERNELS) / sizeof(KERNELS[0]);