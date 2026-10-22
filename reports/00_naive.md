# Kernel NN: naive kernel

**File:** `src/kernels/00_naive.cu`

**Previous kernel:** NN-1 (<None>)

**Date:*10-22-26*

## 1. Change from the previous kernel

<This is the very first kernel.>

## 2. Why it should help

<No bottleneck so far based on previous kernel>

## 3. Prediction (written before profiling)

I have no idea for this one since it's first one 

| Metric (ncu, 1024³) | Previous kernel | Predicted | Reasoning |
|---|---|---|---|
| Duration | | ~ | |
| Busiest unit | | | |
| L1/TEX throughput % | | | |
| DRAM throughput % | | | |
| Sectors / request (global loads) | | | |
| Top stall reason | | | |

## 4. Correctness

| Size | Pass | max_abs_err | max_rel_err |
|---|---|---|---|
| 1000×777×513 | | | |
| 1024³ | | | |
| 2048³ | | | |
| 4096³ | | | |

## 5. Performance (harness, boost clocks)

| Size | ms | TFLOP/s | % cuBLAS | Speedup vs previous |
|---|---|---|---|---|
| 1000×777×513 | 2.1978 | 0.363|9.35 | |
| 1024³ | 10.8529 | 0.198 | 5.21| |
| 2048³ | 94.0203 | 0.183| 3.09 | |
| 4096³ | 891.5345 |0.154 | 3.02 | |
| 8192³ | 6805.9307 |0.162| 5.44  | |


Ceilings for reference: 779 GB/s, 28.4 TFLOP/s.

## 6. Profile: the four questions (ncu, 1024³)

Report: `profiles/NN_1024.ncu-rep`, compared against `profiles/NN-1_1024.ncu-rep` as the baseline.

| Metric | Previous | This kernel | Change |
|---|---|---|---|
| Duration (ms, base clock) | | | |
| Compute (SM) throughput % | | | |
| Memory throughput % | | | |
| L1/TEX throughput % | | | |
| L2 throughput % | | | |
| DRAM throughput % | | | |
| L1 hit rate % | | | |
| L2 hit rate % | | | |
| Sectors / request (loads) | | | |
| Achieved occupancy % | | | |
| Registers / thread | | | |
| Top stall reason | | | |

1. **Which unit is busiest?**
2. **Full (throughput-bound) or waiting (latency-bound)?**
3. **Why?** (the section and number that explains it)
4. **Which line of code?** (from the Source tab)

## 7. Prediction vs result

<What matched, what didn't, and why. The misses are the most useful
part of this section.>

## 8. Next bottleneck

<What limits this kernel now, which becomes the target of the next kernel.>

## 9. Comparison with the reference implementation

<Read the blog's code after yours passed. Differences, and whether they
matter for speed or are just style.>

## 10. Notes

- Logic peeks while writing:
- Bugs hit and how they were found:
- Open questions: