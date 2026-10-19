#include "bench.cuh"
#include <iostream>
using namespace std;

int main() {
    float* A = new float[10];

    fill_random(A, 10, 32323);

    // Use A and B here.
    for(int i = 0; i<10; i++){
        cout << A[i] << endl;
    }

    delete[] A;
    return 0;
}