# Experimental CUDA retrieval and placement

The CPU admits an immutable snapshot, filters typed metadata in row order, and
produces candidate RowIndices. CUDA scores those slots against a resident
embedding allocation. The CPU validates returned score/status pairs, maps
slots to RecordIDs, and builds deterministic Top-K. The CPU comparison uses
the trusted VectorForge executor.

## Contracts and residency

- **Contract S** preserves the CPU executor's observable floating-point
  environment, math provider, `errno`, validation and failure ordering.
  H0's strict CPU route runs before the CUDA module is loaded.
- **Contract G** requires explicit consent. For finite FP32 inputs, each slot
  starts at +0 and uses round-to-nearest FMA in dimension order with gradual
  underflow. A non-finite intermediate records its first dimension. Retrieval
  rejects the first failing slot in candidate order before publishing Top-K.
  CPU floating-point flags, dynamic rounding, math-provider substitution,
  and completed-prefix behavior remain part of S.
- **Residency** owns one private stream, device buffers, and pinned staging.
  The placement, Prepared and mixed owners bind the source snapshot and check
  device, context, and thread ownership. Nonempty direct queries reuse buffer
  capacity and send their own indices and vectors. Empty candidate lists
  submit no copies or kernel. Completion is checked before staging reuse and
  release. Work-submission failures propagate; the CPU alternative is a
  separate decision made before execution. Unconfirmed stream completion or
  safe context ownership at cleanup causes exit 74 before freeing in-use buffers.
- **Memory** for direct scoring is `4ND + 12C + 4D` explicit device bytes for
  N rows, D dimensions, and fixed capacity C. The limits are 128 MiB device,
  4 MiB pinned, and at least 512 MiB free device memory before residency.
  CUDA context overhead is separate. Prepared's additional buffers count
  toward the same budget.

H0 automatic placement is frozen to the historical `N100K-S90` resident session
with eight queries. Eligibility covers the snapshot owner, predicates, query
bits and order, K, batch size, host/device environment, and usable residency.
V2 experiment modes select paths explicitly.

## Build

Use an SM 8.6 GPU and CUDA Toolkit 12.4 or a compatible newer toolkit. The
verified configuration used GCC 13.4, GCC 11.4 as the NVCC host compiler,
CUDA 12.4.131, NVIDIA driver 621.01, and an RTX 3060 Laptop GPU under WSL2.
H0 placement additionally checks its frozen host, driver and runtime
configuration. The standalone build needs the CudaOperator header
`include/cudaHelper/cuda_check.cuh` from a local checkout. The verified header
is from CudaOperator commit `8aed4e271b7d319c45ce90f38027302ece121354`.

Use Bash from the VectorForge root with `CUDA_ROOT` and `CUDA_OPERATOR_ROOT`
set to the toolkit and CudaOperator checkout. Commands use external scratch
directories. Ninja is the build backend for these examples.

```bash
set -e
: "${CUDA_ROOT:?Set CUDA_ROOT to the CUDA toolkit directory}"
: "${CUDA_OPERATOR_ROOT:?Set CUDA_OPERATOR_ROOT to the CudaOperator checkout}"
vf=$(pwd -P)
work=$(mktemp -d)
mkdir "$work/output"

configure_cuda() {
  local build=$1
  shift
  cmake -S "$vf/examples/experimental/cuda_scoring" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="$(command -v g++-13)" \
    -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
    -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++-11)" \
    -DCUDA_OPERATOR_INCLUDE_DIR="$CUDA_OPERATOR_ROOT/include" "$@"
}
```

| Target | Standalone CMake option |
| --- | --- |
| `vectorforge_cuda_scoring_pilot` | Built by default in this CUDA subproject |
| `vectorforge_cuda_resident_retrieval` | `VECTORFORGE_BUILD_RESIDENT_RETRIEVAL` |
| `vectorforge_resident_placement` and its private module | `VECTORFORGE_BUILD_RESIDENT_PLACEMENT` |
| `vectorforge_cuda_locality` | `VECTORFORGE_BUILD_LOCALITY_STUDY` |
| `vectorforge_cuda_prepared` | `VECTORFORGE_BUILD_PREPARED_STUDY` |
| `vectorforge_cuda_mixed_retrieval` | `VECTORFORGE_BUILD_MIXED_RETRIEVAL` |

All listed options default to OFF. Each V2 target can be enabled by itself.
Prepared includes the locality source for its fixture. Locality and mixed reuse
the private retrieval helpers. Each target builds its own executable with only
its private scoring definitions. The root CPU build is separate.

## H0 correctness and bounded demonstrations

The independent CPU workload exports expectations from an AoS full-sort
oracle. The exported files and raw GPU score records use new output paths.

```bash
configure_cuda "$work/h0" \
  -DVECTORFORGE_BUILD_RESIDENT_RETRIEVAL=ON \
  -DVECTORFORGE_BUILD_RESIDENT_PLACEMENT=ON
cmake --build "$work/h0" -j1

cmake -S "$vf/examples/integration/ai_retrieval" -B "$work/consumer" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$(command -v g++-13)" \
  -DVECTORFORGE_SOURCE_DIR="$vf" -DVECTORFORGE_BUILD_AI_RETRIEVAL_WORKLOAD=ON
cmake --build "$work/consumer" -j1
ctest --test-dir "$work/consumer" --output-on-failure

pilot="$work/h0/vectorforge_cuda_scoring_pilot"
retrieve="$work/h0/vectorforge_cuda_resident_retrieval"
place="$work/h0/vectorforge_resident_placement"
s90="$work/output/N100K-S90.expected"

"$pilot" --test
"$pilot" --smoke
"$retrieve" --fixtures
for cell in N100K-S1 N100K-S90; do
  golden="$work/output/$cell.expected"
  "$work/consumer/vectorforge_ai_retrieval_workload" "$cell" --export "$golden"
  "$retrieve" --gate "$cell" "$golden" "$work/output/$cell.bits"
done
"$retrieve" --lifecycle-check N100K-S90 "$s90" > "$work/output/h0-lifecycle-check.jsonl"
"$place" --cpu-tests
"$place" --gpu-tests "$s90" "$work/output/placement-S90.bits"
```

Keep `libvectorforge_placement_cuda.so` beside the placement executable.
Exit zero and `PASS` from the pilot, gates, and placement checks indicate
success. `--smoke` is a small correctness demo without performance sampling.
The lifecycle check records exactly checked setup, session, and cleanup
observations. The H0 golden exports have these SHA-256 digests.

| Expectations | SHA-256 |
| --- | --- |
| `N100K-S1.expected` | `abca42a096e3dc980b35d99cd81e6aa5417043fe3cceb716dae2fe983583dfa8` |
| `N100K-S90.expected` | `9ff395a5e1281f1dff172b7f9f5a19229e0c69c69bdc7a3f2e5fc7818c2e0e6d` |

The following commands time one resident retrieval and one historical
placement session. Each produces query and session records on stdout.

```bash
"$retrieve" --lifecycle N100K-S90 "$s90" resident 8 > "$work/output/h0-resident.jsonl"
"$place" --measure "$s90" gpu > "$work/output/h0-placement.jsonl"
```

H0-06 reproduced the direction of the historical S90/Q8 CPU versus GPU
comparison in two windows. Paired median CPU/GPU ratios were 3.57× and 3.88×
on the recorded device and host. The measured resident interval charged
selection and first query. CUDA context creation, host packing, and the
original embedding upload were recorded separately.

## V2 research results

| Milestone | Question | Result | Engineering decision |
| --- | --- | --- | --- |
| V2-01 | Does physical candidate layout affect scoring? | **C.** Complete clustered retrieval was faster on both routes, with inconsistent GPU-event replay pairs. | Separate first-query effects from warmed replay. |
| V2-02 | Does the layout effect survive warm-up? | **A.** Four warmed pairs favored clustered rows. The paired median scattered/clustered ratio of kernel-event sums was 1.276× on the fixed workload. | Explore preparation cost. Cache and DRAM mechanisms remain unverified. |
| V2-03 | Does compact candidate preparation amortize? | **C.** The inclusive comparison across Q=1, 8, 32, 64, 128 established no reliable crossing. The per-process sum of preparation and incremental cleanup had a median of 4.384 ms. Prepared used 4.92 MiB extra explicit device memory. | Keep Prepared as a fixed-candidate experiment; retain direct scoring for the mixed workload. |
| V2-04B | Can one resident dataset serve changing candidates? | **B.** Exact mixed-query execution passed. All four GPU sessions were shorter in their pairs, with 2.431× GPU/GPU control variation. | Retain the experiment and document session-level timing uncertainty. |

### Locality and warmed replay

The gate for each layout checks independent Top-K, score/status bits, and
layout identity. Successful gate records include `"pass":true`. Replay commands
below each time one process session. `--replay-events` keeps the original replay
boundary. `--replay-warmed` records first query, explicit warm-up, and eight warmed
queries separately.

```bash
configure_cuda "$work/locality" -DVECTORFORGE_BUILD_LOCALITY_STUDY=ON
cmake --build "$work/locality" --target vectorforge_cuda_locality -j1
locality="$work/locality/vectorforge_cuda_locality"
for layout in clustered scattered; do
  "$locality" --gate "$layout" > "$work/output/locality-gate-$layout.jsonl"
  "$locality" --replay-events "$layout" > "$work/output/locality-events-$layout.jsonl"
  "$locality" --replay-warmed "$layout" > "$work/output/locality-warmed-$layout.jsonl"
done
```

### Prepared residency

`--gate` checks compact embedding, score and status bits, Top-K, and
ownership. `--budget` checks aggregate-cap rejection. Both exit zero on
success. The bounded demonstration times one eight-query Direct and one
Prepared session. Prepared separately reports gather, mapping, and release.

```bash
configure_cuda "$work/prepared" -DVECTORFORGE_BUILD_PREPARED_STUDY=ON
cmake --build "$work/prepared" --target vectorforge_cuda_prepared -j1
prepared="$work/prepared/vectorforge_cuda_prepared"
"$prepared" --gate > "$work/output/prepared-gate.jsonl"
"$prepared" --budget > "$work/output/prepared-budget.log"
"$prepared" --direct-g 8 > "$work/output/prepared-direct.jsonl"
"$prepared" --prepared-g 8 > "$work/output/prepared-compact.jsonl"
```

### Mixed-candidate retrieval

The workload varies predicates and vectors across eight queries on one immutable
snapshot. The fixed capacity is 34,000. Candidate counts are 1,000, 33,000,
33,000, 114, 33,000, 34,000, 0, and 1,000 in order. The two adjacent 33,000-slot
sets are disjoint. The owner continues after the zero-candidate query.

The gates check independent exact results and failure handling. Successful
runs emit a final `"pass":true` record. Derive the timing manifest with Python 3
from the independent CPU gate output. Timing modes verify Top-K against this
manifest after the timed interval.

```bash
configure_cuda "$work/mixed" -DVECTORFORGE_BUILD_MIXED_RETRIEVAL=ON
cmake --build "$work/mixed" --target vectorforge_cuda_mixed_retrieval -j1
mixed="$work/mixed/vectorforge_cuda_mixed_retrieval"
"$mixed" --gate-cpu > "$work/output/mixed-gate-cpu.jsonl"
"$mixed" --gate-g > "$work/output/mixed-gate-gpu.jsonl"
"$mixed" --boundaries-g > "$work/output/mixed-boundaries.log"

python3 - "$work/output/mixed-gate-cpu.jsonl" "$work/output/mixed.expected" <<'PY'
import json
from pathlib import Path
import sys

records = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
rows = [row for row in records if row["kind"] == "verified_query"]
gate = records[-1]
if ([row["q"] for row in rows] != list(range(8)) or gate["kind"] != "gate"
        or not gate["pass"] or gate["device"]):
    raise SystemExit("mixed CPU correctness gate did not complete")
keys = ("m", "predicate_fp", "vector_fp", "candidates_fp", "scored_fp", "top_fp")
with Path(sys.argv[2]).open("x") as out:
    for row in rows:
        out.write(" ".join(str(row[key]) for key in keys) + "\n")
PY
```

The next commands time one CPU and one GPU resident session, then collect one
separate GPU diagnostic session. The CPU mode in this standalone executable
does no CUDA device work. Default CPU library and consumer builds omit CUDA.

```bash
"$mixed" --cpu-s "$work/output/mixed.expected" > "$work/output/mixed-cpu.jsonl"
"$mixed" --gpu-g "$work/output/mixed.expected" > "$work/output/mixed-gpu.jsonl"
"$mixed" --diagnostic-g "$work/output/mixed.expected" > "$work/output/mixed-stages.jsonl"
```

### Targeted device checks

After building the corresponding targets above, these run the mixed GPU gate
under Compute Sanitizer. Successful runs report zero errors and, for memcheck,
zero leaked bytes.

```bash
"$CUDA_ROOT/bin/compute-sanitizer" --tool memcheck --leak-check full \
  --error-exitcode 88 "$mixed" --gate-g > "$work/output/mixed-memcheck.log" 2>&1
"$CUDA_ROOT/bin/compute-sanitizer" --tool initcheck \
  --error-exitcode 88 "$mixed" --gate-g > "$work/output/mixed-initcheck.log" 2>&1
```

## Evidence and interpretation

Source and gates reproduce the workloads and generate new local observations.
The original process-pair data, control runs, SASS, and tool logs are retained
in external SHA-256-sealed archives and are outside this repository. Comparing
against those historical raw samples requires access to the corresponding
archive. The single-session demonstrations here produce diagnostic output.
The historical comparisons used separate fresh-process pairs and controls.

CUDA event elapsed time spans same-stream markers and can include gaps between
host submissions. It provides timing evidence within the measured workload and
environment. Direct hardware cache and DRAM mechanisms, portable speedups, and a
CPU/GPU threshold require separate evidence.
