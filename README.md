# blackwell-gemm

Matrix multiplication (FP32 SGEMM) written from scratch in CUDA, optimized one step at a time and measured against cuBLAS on an RTX 5080 Laptop GPU.

## Kernels

| # | Kernel | What changed |
|---|---|---|
| 00 | Naive | One thread per output element, reading straight from global memory. |
| 01 | Coalesced | Neighboring threads read neighboring memory, so loads get batched. |
| 02 | Shared-memory tiling | Each block copies a tile of A and B into shared memory and reuses it. |
| 03 | 1D thread tiling | Each thread computes 8 outputs instead of 1 and keeps its sums in registers. |
| 04 | 2D thread tiling | Each thread computes an 8x8 patch, reusing every loaded value 8 times. |
| 05 | float4 loads | Loads four floats at a time and stores A transposed so shared-memory reads are vectorized too. |
| 06 | Warp tiling | Splits each block's tile between warps, then between the threads of each warp. (this was the hardesst one : (  ))|

Code is in `src/kernels/`, one file per kernel.

## Results

<img src="results/plots/rtx5080_laptop_ladder_light.png" alt="Kernel performance against cuBLAS on RTX 5080 Laptop" width="600">

## Run it

Needs an NVIDIA GPU, the CUDA toolkit and CMake.

```bash
cmake -S . -B build && cmake --build build -j
./build/gemm              # every kernel, every size
./build/gemm 6            # one kernel
```

Every kernel is checked against cuBLAS before it is timed.
