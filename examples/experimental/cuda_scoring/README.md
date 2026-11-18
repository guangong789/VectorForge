# Experimental CUDA retrieval and placement

This standalone experiment measures resident GPU scoring in an end-to-end
retrieval workload. The CPU filters metadata in row order, CUDA scores the
candidate rows against resident embeddings, and the CPU maps scores to record
IDs and constructs deterministic Top-K results. The CPU baseline calls the
unchanged trusted VectorForge executor. The lifecycle modes measure the effect
of CUDA initialization, embedding upload, and reuse on total query cost.

## Numerical and placement contracts

- **Contract S** preserves the production CPU executor's observable behavior,
  including its floating-point environment, math provider, and validation and
  failure ordering. The placement host takes this path before loading a CUDA
  module or querying a device.
- **Contract G** requires explicit consent. It uses finite FP32 inputs, an
  initial positive zero, one round-to-nearest FMA per dimension in order,
  gradual underflow, and a finite check after each step. The result rejects a
  numerical failure instead of returning a partial Top-K. G excludes equivalence
  of CPU FP flags, dynamic rounding, `errno`, math-provider substitution,
  completed-prefix behavior, and allocation and exception precedence.

Automatic placement is experimental and covers exactly the historical
`N100K-S90` resident session with eight queries. It requires matching dataset
provenance, predicates, query bits and order, a verified device and context, and
a valid private resident owner. The current placement implementation also
checks the measured WSL2 kernel and CPU model. This standalone consumer exposes
no production GPU backend, generic cost predictor, online scheduler, graph, or
MLIR integration.

## Build

The verified environment is Ubuntu 22.04 under WSL2 with an RTX 3060 Laptop GPU
(SM 8.6), NVIDIA driver 621.01, CUDA Toolkit 12.4.131, GCC 13.4 for C++, GCC
11.4 as the NVCC host compiler, CMake 4.2.0, and Ninja 1.10.1. These versions
identify the tested configuration. The CUDA targets use C++20, target SM 8.6,
and link the shared CUDA Runtime. The standalone CMake project requires a
CudaOperator checkout supplying `include/cudaHelper/cuda_check.cuh`. The
verified header is from CudaOperator commit
`8aed4e271b7d319c45ce90f38027302ece121354`. No CudaOperator build, PyTorch, or
external evidence archive is needed.

From the VectorForge checkout, export `CUDA_OPERATOR_ROOT` and `CUDA_ROOT`
to the CudaOperator checkout and CUDA Toolkit installation directories. Run
the commands below in Bash. All build products and test output go to the
temporary workspace.

```bash
vf=$(pwd -P)
work=$(mktemp -d)
mkdir "$work/expectations"
cxx=$(command -v g++-13)
cuda_host=$(command -v g++-11)

cmake -S "$vf/examples/experimental/cuda_scoring" -B "$work/gpu" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_COMPILER="$cxx" \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="$cuda_host" \
  -DCUDA_OPERATOR_INCLUDE_DIR="$CUDA_OPERATOR_ROOT/include" \
  -DVECTORFORGE_BUILD_RESIDENT_RETRIEVAL=ON \
  -DVECTORFORGE_BUILD_RESIDENT_PLACEMENT=ON
cmake --build "$work/gpu" -j1

cmake -S "$vf/examples/integration/ai_retrieval" -B "$work/consumer" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$cxx" \
  -DVECTORFORGE_SOURCE_DIR="$vf" -DVECTORFORGE_BUILD_AI_RETRIEVAL_WORKLOAD=ON
cmake --build "$work/consumer" -j1
ctest --test-dir "$work/consumer" --output-on-failure

cmake -S "$vf" -B "$work/cpu" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$cxx"
cmake --build "$work/cpu" -j1
ctest --test-dir "$work/cpu" --output-on-failure
```

The CUDA build produces `vectorforge_cuda_scoring_pilot`,
`vectorforge_cuda_resident_retrieval` and `vectorforge_resident_placement`.
Keep `libvectorforge_placement_cuda.so` beside the placement executable. Only
the experimental G residency preparation loads that module. The default
VectorForge build uses the CPU alone.

## Correctness and lifecycle commands

Generate local expectations with the CPU consumer's independent full-sort
oracle. The workload generator uses the fixed seed `0x56463112C`, and both cells
use the same eight ordered query vectors. The export command requires a new
output path. Fresh exports can be checked against these historical SHA-256
digests.

| Expectations file | SHA-256 |
| --- | --- |
| `N100K-S1.expected` | `abca42a096e3dc980b35d99cd81e6aa5417043fe3cceb716dae2fe983583dfa8` |
| `N100K-S90.expected` | `9ff395a5e1281f1dff172b7f9f5a19229e0c69c69bdc7a3f2e5fc7818c2e0e6d` |

```bash
pilot="$work/gpu/vectorforge_cuda_scoring_pilot"
retrieve="$work/gpu/vectorforge_cuda_resident_retrieval"
place="$work/gpu/vectorforge_resident_placement"
s90="$work/expectations/N100K-S90.expected"

for cell in N100K-S1 N100K-S90; do
  golden="$work/expectations/$cell.expected"
  "$work/consumer/vectorforge_ai_retrieval_workload" "$cell" --export "$golden"
  sha256sum "$golden"
done

"$pilot" --smoke
"$pilot" --test
"$retrieve" --fixtures
for cell in N100K-S1 N100K-S90; do
  "$retrieve" --gate "$cell" "$work/expectations/$cell.expected" "$work/$cell.bits"
done
"$retrieve" --lifecycle-check N100K-S90 "$s90"
"$place" --cpu-tests
"$place" --gpu-tests "$s90" "$work/placement-S90.bits"

"$CUDA_ROOT/bin/compute-sanitizer" --target-processes all --tool memcheck \
  --error-exitcode 88 --leak-check full "$place" --gpu-tests "$s90"
"$CUDA_ROOT/bin/compute-sanitizer" --target-processes all --tool initcheck \
  --error-exitcode 88 "$place" --gpu-tests "$s90"
```

## Performance and evidence

Run fresh processes in balanced order to measure the approved placement case.
The `gpu` arm explicitly accepts Contract G. Its timed interval includes the
placement decision, query validation, first query, result consumption and
destruction. Initial context setup, packing and residency preparation are
recorded separately. The CPU arm consumes the same eight owned results.

```bash
for path in cpu gpu gpu cpu cpu gpu gpu cpu; do
  "$place" --measure "$s90" "$path"
done
```

Historical observations used the hardware and software above. Both workloads
have 100,000 rows, dimension 128 and K=20. S1 matches 1,000 rows and S90 matches
90,000. The figures describe this workload and measurement configuration.

- **H0-02** used four CPU/GPU pairs per cell after query warm-up. The median
  eight-query CPU/resident GPU times were 3.979/3.148 ms for S1 and
  98.633/24.846 ms for S90.
- **H0-03** used four independent sessions per workload and session length,
  including the first query. S1 already resident crossed between 8 and 32
  queries. S90 with a warm runtime and fresh residency crossed between 1 and 8.
  S90 cold-start results at 32 and 64 queries had mixed pair outcomes. The
  repeatable cold-start bracket was only (8, 128].
- **H0-05** used four balanced pairs for S90 resident Q=8 with the placement
  decision and first query charged. CPU/GPU medians were 102.26/26.37 ms,
  the median paired CPU/GPU ratio was 3.88×, and the decision median was 4.22 µs.

Local exports and the commands above rebuild correctness and measurement
results. Verifying historical samples requires the separately retained evidence
archives and their `evidence-file-hashes.json` manifests. Those archives and
their raw timings are outside Git.

## Resource and failure boundaries

The private, single-threaded owner retains the sealed CPU snapshot used to
upload embeddings. It holds one stream and its device and pinned staging
buffers. It checks upload completion and verifies the current context before
use. Scoring synchronizes copies and kernels before staging reuse or release.
Explicit allocation limits are 128 MiB on device and 4 MiB pinned on host.
New residency requires at least 512 MiB free device memory. S90 used 49.97 MiB
of explicit device memory and 1.15 MiB pinned on host. These counts exclude
CUDA context overhead.

A closed or invalid session is ineligible. The caller may authorize a CPU
alternative before execution begins. CUDA errors after
submission propagate without a CPU retry. If the experiment cannot confirm
stream completion or safe context ownership at cleanup, it exits with status
74 without releasing buffers that may still be in use.
