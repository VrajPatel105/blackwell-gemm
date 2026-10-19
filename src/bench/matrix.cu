
#include <cuda.h>
#include <iostream>
#include <random>
#include <cfloat>
#include <cmath>
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

    double rtol = K * FLT_EPSILON;
    double atol = rtol;

    double max_abs_err = 0.0; // abs(x_measured - x_true)
    double max_rel_err = 0.0; // max_abs_err / x_true
    bool flag_so_far = true;

    for(size_t i = 0; i < size_t(M)*N; i++){

        double curr_ele_abs_err = fabs(out[i] - ref[i]);

        if (curr_ele_abs_err > max_abs_err) max_abs_err = curr_ele_abs_err;
        
        // skipping the perfect 0 condition from reference so that we dont get ele_rel_err to infinite and then for it to report a inf for a perfect correct kernel.
        if (ref[i] != 0.0f) {
            double curr_ele_rel_err = curr_ele_abs_err / fabs(ref[i]);
            if (curr_ele_rel_err > max_rel_err) max_rel_err = curr_ele_rel_err;
        }

        if(curr_ele_abs_err > max_abs_err){
            max_abs_err = curr_ele_abs_err;
        }

        if(!(curr_ele_abs_err <= (atol + rtol * fabs(ref[i])))){
            flag_so_far = false;
        }

    }

    return CheckResult{max_abs_err, max_rel_err, flag_so_far};

}