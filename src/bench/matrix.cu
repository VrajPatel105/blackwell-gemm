
#include <cuda.h>
#include <iostream>
#include <random>
#include 
#include "bench.cuh"

using namespace std;


__host__ void fill_random(float* h, size_t n, unsigned seed){
    mt19937 mt(seed); // using the Mersenne Twister algo instead of rand one.
    uniform_real_distribution<double> distribution(-1, 1);

    for(size_t i = 0; i < n; i++){
        h[i] = distribution(mt);
    }

}

__host__ CheckResult compare(const float* ref, const float* out, int M, int N, int K){
    // this func is direct implementation to numpy.allclose func. 
    // I did not want numpy to be a dependency for the project, so I decided to implement the logic here itself.

    // numpy.allclose() returns true if this is true : absolute(a - b) <= (atol + rtol * absolute(b))
    // b is the cublass one. h_ref

    double rtol = K * 


}