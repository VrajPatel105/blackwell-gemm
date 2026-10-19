#pragma once
#include <cublas_v2.h>
#include <cstddef>

struct CheckResult {double max_abs_err; double max_rel_err; bool pass; };

void fill_random(float* h_buffer, size_t n, unsigned seed);
CheckResult compare(const float* h_ref, const float* h_out, int M, int N, int K);
void cublas_sgemm(cublasHandle_t handler, int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C);
