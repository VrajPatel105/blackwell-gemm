#!/usr/bin/env bash
# One command: build, GPU info, ceilings, benchmark, plots, ncu profile per kernel, nsys trace, bundle.
#
#   scripts/run_all.sh <tag> "<GPU name>" [cuda_arch]
#   scripts/run_all.sh b200 "NVIDIA B200" 100
#   scripts/run_all.sh dryrun "RTX 5080 Laptop" 120
#
# Outputs:
#   results/<tag>.csv, <tag>.md, <tag>_gpu.txt, <tag>_ceilings.txt, plots/<tag>_*.png
#   profiles/<tag>/NN_<name>_<size>.ncu-rep   one per kernel
#   profiles/<tag>/nsys_all.nsys-rep
#   <tag>_bundle.tar.gz                       everything above, ready to scp
#
# Optional: PROFILE_SIZE=4096 (matrix size for ncu and nsys), SKIP_BENCH=1, SKIP_NCU=1, SKIP_NSYS=1
set -uo pipefail

TAG="${1:?usage: scripts/run_all.sh <tag> \"<GPU name>\" [cuda_arch]}"
GPU_NAME="${2:?usage: scripts/run_all.sh <tag> \"<GPU name>\" [cuda_arch]}"
ARCH="${3:-}"
PROFILE_SIZE="${PROFILE_SIZE:-4096}"

cd "$(dirname "$0")/.."
RES="results"
PROF="profiles/$TAG"
mkdir -p "$RES" "$PROF"

step() { printf '\n==> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; WARNINGS+=("$*"); }
WARNINGS=()

find_tool() {  # find_tool <name> <extra glob>...
  local name="$1"; shift
  if command -v "$name" >/dev/null 2>&1; then command -v "$name"; return 0; fi
  local p
  for p in "$@"; do [ -x "$p" ] && { echo "$p"; return 0; }; done
  return 1
}

# ---------------------------------------------------------------- 1. build
step "Building (arch: ${ARCH:-from CMakeLists})"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release ${ARCH:+-DCMAKE_CUDA_ARCHITECTURES=$ARCH} >/dev/null \
  && cmake --build build --target gemm ceilings -j \
  || { echo "build failed, stopping"; exit 1; }

# ---------------------------------------------------------------- 2. GPU info
step "Recording GPU info"
{
  echo "tag: $TAG"
  echo "gpu: $GPU_NAME"
  date -u +"date: %Y-%m-%d %H:%M UTC"
  echo
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,temperature.gpu,clocks.max.graphics --format=csv
  echo
  nvcc --version | tail -n 2
  echo
  nvidia-smi
} > "$RES/${TAG}_gpu.txt" 2>&1
head -n 6 "$RES/${TAG}_gpu.txt"

# ---------------------------------------------------------------- 3. ceilings
step "Measuring ceilings (bandwidth, peak FP32)"
./build/ceilings | tee "$RES/${TAG}_ceilings.txt" || warn "ceilings run failed"

# ---------------------------------------------------------------- 4. benchmark
if [ -z "${SKIP_BENCH:-}" ]; then
  step "Benchmarking every kernel at every size"
  ./build/gemm all --csv > "$RES/$TAG.csv" || warn "a kernel failed its correctness check, see $RES/$TAG.csv"
  column -s, -t "$RES/$TAG.csv" | cut -c1-150

  step "Plotting"
  python3 -c "import matplotlib" 2>/dev/null || python3 -m pip install -q matplotlib 2>/dev/null \
    || python3 -m pip install -q --break-system-packages matplotlib
  python3 scripts/plot_results.py "$RES/$TAG.csv" --gpu "$GPU_NAME" || warn "plotting failed"
fi

# ---------------------------------------------------------------- 5. ncu
if [ -z "${SKIP_NCU:-}" ]; then
  step "Nsight Compute: one profile per kernel at ${PROFILE_SIZE}^3"
  if NCU="$(find_tool ncu /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu)"; then
    # performance counters are often admin-only: try plain first, then sudo
    RUN_NCU=("$NCU")
    if ! "$NCU" --set full -o "$PROF/_permtest" -f ./build/gemm 0 --profile 256 >"$PROF/_permtest.log" 2>&1; then
      if grep -q ERR_NVGPUCTRPERM "$PROF/_permtest.log"; then
        echo "counters need admin rights, switching to sudo"
        RUN_NCU=(sudo env "PATH=$PATH" "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" "$NCU")
      fi
    fi
    rm -f "$PROF"/_permtest*

    ids=$(./build/gemm all --size 64 --csv 2>/dev/null | awk -F, 'NR>1 {print $1 ":" $2}')
    for entry in $ids; do
      id="${entry%%:*}"; name="${entry#*:}"
      out="$PROF/$(printf '%02d' "$id")_${name}_${PROFILE_SIZE}"
      echo "  kernel $id ($name)"
      "${RUN_NCU[@]}" --set full -f -o "$out" ./build/gemm "$id" --profile "$PROFILE_SIZE" >"$out.log" 2>&1 \
        || warn "ncu failed for kernel $id, see $out.log"
    done
  else
    warn "ncu not found; install the nsight-compute package and re-run with SKIP_BENCH=1"
  fi
fi

# ---------------------------------------------------------------- 6. nsys
if [ -z "${SKIP_NSYS:-}" ]; then
  step "Nsight Systems: timeline of all kernels at ${PROFILE_SIZE}^3"
  if NSYS="$(find_tool nsys /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys)"; then
    "$NSYS" profile --trace=cuda --force-overwrite true -o "$PROF/nsys_all" \
      ./build/gemm all --size "$PROFILE_SIZE" >"$PROF/nsys_all.log" 2>&1 \
      || warn "nsys failed, see $PROF/nsys_all.log"
  else
    warn "nsys not found; install the nsight-systems package and re-run with SKIP_BENCH=1 SKIP_NCU=1"
  fi
fi

# ---------------------------------------------------------------- 7. bundle
step "Bundling"
sudo chown -R "$(id -u):$(id -g)" "$PROF" 2>/dev/null || true
BUNDLE="${TAG}_bundle.tar.gz"
tar czf "$BUNDLE" "$PROF" $(ls -d "$RES/$TAG".* "$RES/${TAG}_"* "$RES/plots/${TAG}_"* 2>/dev/null)
ls -lh "$BUNDLE"
echo "ncu reports: $(ls "$PROF"/*.ncu-rep 2>/dev/null | wc -l)   nsys reports: $(ls "$PROF"/*.nsys-rep 2>/dev/null | wc -l)"

if [ "${#WARNINGS[@]}" -gt 0 ]; then
  printf '\n%d warning(s):\n' "${#WARNINGS[@]}"; printf '  - %s\n' "${WARNINGS[@]}"
fi
echo
echo "Copy it to your machine (run this on your laptop):"
echo "  scp $(whoami)@<server-ip>:$(pwd)/$BUNDLE ."
