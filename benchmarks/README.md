# Reproducible execution-path baseline

Build and run from the repository root (no additional dependencies):

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DVECTORFORGE_BUILD_BENCHMARKS=ON
cmake --build build-release -j 2
ctest --test-dir build-release --output-on-failure
./build-release/vectorforge_benchmark > build-release/benchmark-baseline.csv
```

The opt-in target is `vectorforge_benchmark`. With a multi-configuration generator,
build with `--config Release` and use the corresponding executable directory.
Debug/unrecognized configurations and sanitizer builds reject performance runs.
`--smoke` explicitly runs a tiny fixed case in any build, labels its CSV `smoke`,
and is suitable for correctness tests only. Fast-math compilation is rejected.

Default sampling is two warm-ups and seven measured calls **per executor per
workload**. Override sampling/seed for the matrix, or any workload parameter to
select one custom case (unspecified parameters retain the first case's defaults):

```sh
./build-release/vectorforge_benchmark --warmup 3 --iterations 11 --seed 1447440437 > build-release/repeat.csv
./build-release/vectorforge_benchmark --rows 20000 --dimension 64 --selectivity 25 --k 100 --batch-size 127 --iterations 9 > build-release/custom.csv
./build-release/vectorforge_benchmark --help
```

Integers use decimal notation; D and B must be positive; selectivity is an integer
percentage from 0 to 100; measured iterations must be at least three. Zero rows,
zero K and zero warm-ups are supported. Huge shapes/overflow are rejected before
allocation; custom large but representable datasets remain the caller's RAM responsibility.

The eight default cases are deliberate single-axis variations, not a Cartesian
product: 10,000/50,000 rows, 32/128 dimensions, 1%/10%/100% selectivity, K=10/10,000,
and B=32/256/2048. The largest default embedding allocation is below 7 MiB; each
dataset is destroyed before generating the next one.

## Reproducibility and timing

- Dataset generation and query construction happen outside timing. `mt19937_64`
  with default seed `0x56463035` generates float32 values in [-1,1] in exact 1/1024
  steps. Query values are generated first, followed by row-major embeddings.
- One int32 filter column holds zeros and ones. Exactly `floor(N * percentage / 100)`
  rows contain one; a separate seeded Fisher–Yates permutation scatters them.
  Every query uses `column == int32_t{1}`. Changing K/B does not change data; changing
  selectivity preserves embeddings/query. IDs descend from N to 1.
- Both executors reuse the same immutable owners, query and predicate. Exact ID,
  order and float32 score-bit equivalence is checked before warm-up and after
  every later call, outside its timing window. Checksums make results observable.
- `steady_clock` measures each synchronous query call from entry through returned
  owning result. This includes input validation and its allocations, scoring,
  ranking and result allocation. Dataset generation, reference comparisons,
  checksum calculation, destruction of returned results, statistics and output
  formatting/I/O are excluded. Normal allocator reuse and warm caches are included.
- Calls alternate reference-first/batched-first each round. Runs are sequential,
  single-threaded, and not CPU-pinned. Background load, WSL scheduling, frequency
  scaling and allocator state can affect timings. Reproducibility means identical
  workload data, semantics and procedure, not identical wall-clock durations.
- CSV contains one summary row per executor/workload: median, min/max, all individual
  samples (semicolon separated), input rows/second (`N / median seconds`, **not**
  qualifying rows/second), requested/actual selectivity, candidate/result counts,
  seed, dataset/result fingerprints, compiler/configuration/flags, CPU, OS,
  process environment and UTC run timestamp. Reference B=0 means not applicable.
  Fingerprints are reproducibility aids, not cryptographic hashes.
- The no-fast-math, sequential float32 FMA contract is unchanged. Compare complete
  execution paths: traversal and Top-K both differ, so timing differences cannot
  be attributed solely to the heap. Preserve slower cases; do not infer universal
  speedups. No allocation or peak-extra-memory claims are made without instrumentation.

## Small smoke tests under sanitizers

```sh
cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug -DVECTORFORGE_ENABLE_SANITIZERS=ON -DVECTORFORGE_BUILD_BENCHMARKS=ON
cmake --build build-sanitize -j 2
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 ctest --test-dir build-sanitize --output-on-failure
```

Run leak-enabled tests through the approved ordinary-user path outside the Codex
sandbox; its ptrace restrictions are already diagnosed. No kernel/seccomp changes
are needed. CTest runs only tiny benchmark smoke cases alongside correctness tests;
never use sanitizer timings as a performance baseline. Raw CSV stays in ignored
build directories, keeping the source tree light.

## Isolated Top-K comparison (V0-06)

The same opt-in build also produces `vectorforge_topk_benchmark`. The existing E2E
executable, options and CSV schema remain unchanged. This standalone experiment
calls two benchmark-local algorithms; it never invokes an executor in its timed
region and does not change either production executor.

```sh
./build-release/vectorforge_topk_benchmark > build-release/topk-baseline.csv
./build-release/vectorforge_topk_benchmark --candidates 50000 --k 500 --distribution ties --warmup 3 --iterations 11 --seed 1447440438 > build-release/topk-custom.csv
./build-release/vectorforge_topk_benchmark --smoke
./build-release/vectorforge_topk_benchmark --help
```

Any of `--candidates`, `--k`, or `--distribution` selects one custom workload; the
defaults for its remaining fields are M=10,000, K=10, distribution=random. Sampling
defaults are two warm-ups and seven measured calls per algorithm; the default
`mt19937_64` seed is `0x56463036` (1447440438). Debug/sanitized performance runs are
rejected; CTest uses only tiny correctness and `--smoke` cases under sanitizers.

The ten default cases cover M=10,000/100,000, K=0/1/10/9,000/10,000/10,001, and four
distributions. This is a selected matrix, not a Cartesian product. Random scores
are finite float32 multiples of 1/1024 from -1,000,000/1024 to +1,000,000/1024.
The `ties` case uses eight integer scores from -4 to 3. Ascending/descending cases
reorder the same random candidate set outside timing, with ID descending/ascending
respectively for score ties. IDs are unique and initially descend M..1. Changing
K leaves candidate generation unchanged. The largest default input is 100,000
records (about 1.6 MB with this platform's 16-byte `ScoredRecord`).

**Primary timing boundary:** transform an already resident, immutable candidate
sequence into an owning, ordered Top-K result. Full Sort includes allocation,
copying **all M** records, full sorting, and truncation to min(K,M). Bounded Heap
includes its allocation/growth, every input-order heap update and final in-place
sorting. Both immediately return an empty vector for K=0 without scanning or
copying; those timings are dominated by clock/call overhead and are not throughput
evidence. Neither implementation special-cases K near M or switches algorithms.

Generation, finite/unique-ID validation, expected results and fingerprints are
prepared outside timing. Every returned result is checked against the expected
ID order and float32 bits after the stop timestamp; serialization, statistics and
returned-vector destruction are also excluded. Algorithm order alternates each
round, with both algorithms sharing exactly the same input owners. There is no
cache flushing, CPU pinning or affinity control. Cache/allocator reuse, background
load and WSL scheduling remain part of the environment; no universal speedup is
claimed.

**Memory boundary:** the M-element input and a min(K,M)-element expected result
are resident before measurement. The temporary correctness-validation ID set and
full-sort preparation workspace are freed before warm-up. During a measured call,
Full Sort uses an M-element candidate copy (and retains its allocation even after
truncation); Heap retains at most min(K,M) records, grows on demand and never
reserves an arbitrarily large K. Vector capacity may exceed its size. These are
algorithm storage boundaries, **not measured peak RSS, streaming peak memory, or
allocation counts**. No copying/scoring/embedding/query work is hidden outside the
stated boundary to favor either algorithm.

Top-K CSV has its own schema: algorithm, M, K, K/M, distribution, warm-up/iterations,
seed, median/min/max, individual samples, returned count, input/result fingerprints,
the same compiler/build/CPU/OS/environment metadata as E2E, and an explicit
`timing_boundary` label. K/M is empty when M=0. Keep winning and losing cases;
these measurements isolate candidate reduction and cannot predict E2E latency by
themselves. Raw baseline CSV belongs in the ignored build directory.

## Validated admission and query reuse (V0-08)

The opt-in build also produces `vectorforge_reuse_benchmark`, using the existing
batched legacy API and `ValidatedDataset` trusted API without altering either.
Existing E2E/Top-K commands and CSV schemas stay unchanged.

```sh
./build-release/vectorforge_reuse_benchmark > build-release/admission-reuse-baseline.csv
./build-release/vectorforge_reuse_benchmark --rows 20000 --dimension 64 --selectivity 25 --k 10 --batch-size 127 --reuse 100 --warmup 3 --iterations 11 --seed 1447440437 > build-release/reuse-custom.csv
./build-release/vectorforge_reuse_benchmark --smoke
./build-release/vectorforge_reuse_benchmark --help
```

Any workload option or `--reuse R` selects one custom case. Unspecified query
fields use the E2E defaults; custom R defaults to 10 and must be positive. Global
sampling/seed options alone retain the default matrix. Debug/sanitized performance
runs are rejected; sanitizer CTest runs only fixed tiny smoke/correctness cases.

Four default datasets vary one axis at a time: N=10,000/D=32/10% selectivity,
N=50,000/D=32/10%, N=10,000/D=128/10%, and N=10,000/D=32/100%. All use K=10/B=256,
the existing deterministic generator/seed `0x56463035`, and two warm-ups/seven
measured iterations per component. Each dataset reports R=1/10/100 using the
**same measured components**, yielding twelve CSV rows, not twelve independently
timed workflows. Inputs remain small enough for a development laptop.

Three components are measured separately with `steady_clock`:

1. **Admission:** construct a fresh `ValidatedDataset` from the unchanged source,
   including the real snapshot copy, row/finiteness checks, and duplicate-ID set
   creation/destruction inside the admission API. Previous snapshot destruction
   happens before the start timestamp; correctness checks happen after the stop.
2. **Legacy query:** one complete legacy batched call, including all its ordinary
   validation, scoring, bounded-heap Top-K and return allocation.
3. **Trusted query:** one complete trusted batched call on the **same retained
   validated instance**. The last measured admission supplies this instance; it
   stays alive across every subsequent query warm-up and sample. No query sample
   constructs, copies or admits another dataset.

Dataset/query generation and an initial legacy result are prepared outside all
timing. Every admission is checked by a trusted query outside its admission timer;
every later query result is checked for exact ID ordering and float32 bits outside
its query timer. Result fingerprints make outputs observable. Legacy-first and
trusted-first query order alternates each round after warm-up. Both the original
source and validated snapshot remain alive throughout query comparisons. The
snapshot owns independent storage; no source aliases are modified. Returned-result
destruction, snapshot retirement, comparisons and output I/O are excluded from the
reported components. Admission samples finish before the query measurement phase.

**Workflow totals and break-even are models, not separately timed workflows:**

```text
legacy_total_model(R)  = R * median_legacy_query
trusted_total_model(R) = median_admission + R * median_trusted_query
```

If trusted queries are faster, the smallest strictly winning positive integer R
is `floor(median_admission / (median_legacy_query - median_trusted_query)) + 1`.
Equality is not a win. Threshold arithmetic uses long double on the measured
binary64 medians; unrepresentable thresholds/totals are reported as errors. If
trusted queries are equally fast or slower, the threshold is empty and
`break_even_status=trusted_not_faster`. The estimate assumes constant per-query
costs and does not model cold-start behavior, drift, cache/allocator changes,
correlated timing variance or dataset retirement. It is workload/environment
specific; it is not a universal reuse threshold.

CSV records N/D/selectivity/K/B/R, qualifying/returned counts, median/min/max and
all raw samples for each measured component, the two `*_total_ms_model` fields,
`total_kind=model_from_component_medians`, `break_even_reuse_model`, status,
data/result fingerprints, sampling/seed and the existing build/machine/environment
metadata. Component min/max are measured variability; no workflow variability or
peak-memory claim is invented from their sum. The source plus its copied snapshot
are resident for query comparisons; no alternative ownership strategy or zero-copy
admission is substituted. Preserve both winning and losing modeled scenarios.

## Sequential similarity kernel (V1-01)

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DVECTORFORGE_BUILD_BENCHMARKS=ON -DVECTORFORGE_SIMILARITY_DIAGNOSTICS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-release -j 2
./build-release/vectorforge_similarity_benchmark > build-release/similarity-baseline.csv
# Custom shape/sampling (decimal seed); sampling flags alone retain the default matrix:
./build-release/vectorforge_similarity_benchmark --rows 4096 --dimension 128 \
  --warmup 2 --iterations 7 --seed 1447440437
objdump -drwC build-release/CMakeFiles/vectorforge_similarity_kernel.dir/benchmarks/similarity_kernel.cpp.o \
  > build-release/similarity-kernel.asm
```

The separate benchmark-only `similarity_kernel.cpp` contains the actual measured
kernel; production executors and ValidatedDataset are unchanged. Each call scans
all N contiguous row-major embeddings with one fixed query, starting each row at
positive zero and applying float32 `std::fma` in feature order. Every intermediate
score must be finite, matching the current execution contract. Timings include
constant shape checks, the FMA loops, finite checks and preallocated score stores;
they exclude allocation, data generation, filtering, Top-K, verification and I/O.
Per-dot nanoseconds are scan time/N, not isolated single-call latency. N=0 is a
correctness boundary with blank per-dot latency, not a performance measurement.

Six default cases pair N=1024/8192 with D=32/128/768; only one dataset is measured
at a time, with at most 24 MiB of embeddings. Fixed-seed `mt19937_64` produces
bounded finite dyadic values in [-1,1], using the existing seed and integer mapping.
Before warm-up, scores are checked bit-for-bit against the unchanged production
scalar reference. Its temporary copied owners/IDs/sorted records are released
before timing. Every later call's entire output is checked and fingerprinted
after the timer; the kernel is compiled separately without CMake IPO. Default
sampling uses two warm-ups and seven scans with steady_clock. No CPU pinning,
cache flush or cold-cache claim; full output verification also warms score arrays.
Debug/sanitizer runs allow only `--smoke` at the CLI and use tiny inputs.

The 27-column CSV records N/D/embedding bytes, median/min/max and all scan samples,
derived dots/s and ns/dot, data/result fingerprints, compile-time kernel ISA macros,
compiler/build flags and the existing CPU/OS/environment/timestamp metadata. These
are measured scans, not query or end-to-end executor costs. No alternate accumulator
or numerical relaxation is used; sorting continues to use the existing exact rules.

With the GCC-only diagnostics option, the **same compiled kernel** emits
`build-release/similarity-vectorization-Release.txt` and
`build-release/CMakeFiles/vectorforge_similarity_kernel.dir/benchmarks/similarity_kernel.cpp.s`.
The option only adds reports/save-temps to that target. Reports may append on
recompilation; use a fresh build directory for a clean capture. See
`compile_commands.json` for the exact command. To record GCC's effective ISA,
use its command's code-generation flags with `c++ -Q --help=target -x c++ -c
/dev/null -o /dev/null`; CPU features alone do not identify the compilation target.

Observed locally with GCC 13.4.0, `-O3 -DNDEBUG -std=c++20`, the existing warning
flags, and default `-march=x86-64 -mtune=generic`: SSE2 enabled, AVX/AVX2/FMA
disabled. GCC reports **0 vectorized loops**, citing control flow in the loop
(the per-step finite check/throw). Object and linked assembly show scalar `movss`,
one `fmaf` library call per feature, and scalar `ucomiss` checking each result.
The `andps` instruction clears sign bits for the finite check; XMM registers or
packed bitwise instructions are not evidence of packed floating-point arithmetic
or a vector reduction. No packed multiply/add/FMA reduction appears in this kernel.
The library's runtime FMA backend was not profiled; a `fmaf` call does not imply
software emulation. Separately, the score-to-score dependency and exact rounding
order prohibit replacing the feature reduction with reassociated partial sums.
The report establishes this build's outcome, not that SIMD across rows is impossible.

Baseline CSV, source/build context, full GCC report, target options and disassembly
stay under ignored `build-release/`. Interpret timings as an unpinned WSL2 laptop
baseline; cache/thermal/scheduling effects and library-call costs limit portability.

## Scalar hardware FMA target comparison (V1-02)

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DVECTORFORGE_BUILD_BENCHMARKS=ON -DVECTORFORGE_BUILD_FMA_COMPARISON=ON \
  -DVECTORFORGE_SIMILARITY_DIAGNOSTICS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-release -j 2
./build-release/vectorforge_fma_benchmark --iterations 8 > build-release/fma-target-baseline.csv
# The same shape/sampling options as the V1-01 benchmark:
./build-release/vectorforge_fma_benchmark --rows 4096 --dimension 128 \
  --warmup 2 --iterations 8 --seed 1447440437
```

This opt-in comparison requires GCC/Clang targeting x86-64. The local i9-12900H
under WSL2 exposes AVX and FMA (also AVX2, which this comparison does not enable).
A baseline-compiled `__builtin_cpu_supports` guard checks usable AVX/FMA before
any hardware-target call, including metadata calls. An unsupported CPU exits 77
without executing that target; CTest marks its two comparison tests skipped.
`--help` remains available. No production ISA dispatch is introduced.

Both static targets compile **the same** `similarity_kernel.cpp`, with private
namespace definitions isolating their symbols from each other and from V1-01:

| Target | Private code-generation options |
| --- | --- |
| `vectorforge_fma_generic_kernel` | `-march=x86-64 -mtune=generic -fno-fast-math -mno-avx -mno-avx2 -mno-fma` |
| `vectorforge_fma_hardware_kernel` | `-march=x86-64 -mtune=generic -fno-fast-math -mavx -mno-avx2 -mfma` |

Both use the same Release `-O3 -DNDEBUG`, C++20 and warning options; CMake IPO is
disabled on these kernels. The harness, original V1-01 target and production
compile options are unchanged. Use empty global `CMAKE_CXX_FLAGS` as in the local
baseline; do not add global native/fast-math/LTO options to reproduce this experiment.
Compiler commands in `compile_commands.json` are the authority for actual flags.

The dataset generator, seed, six N/D shapes, one-scan timing boundary and output
checks follow V1-01. Both targets read the identical resident query/embedding
arrays and write the same preallocated score array. The original V1-01 kernel
establishes expected scores outside timing; both target outputs must match every
score bit before warm-up and after **every** timed call. Separate tests also
compare both targets directly to the unchanged production reference, using nine
dimensions and three seeds, five repetitions, cancellation/order-sensitive
values, signed zeros, subnormals/gradual underflow, near-maximum finite FMA and
intermediate-overflow rejection. This is tested compatibility, not an exhaustive
proof for all floating-point environments or inputs.

Each warm-up/measured round runs both targets and alternates which runs first,
starting with generic. The recorded run uses two warm-ups and eight measured
rounds: each target runs first four times. Default CLI sampling remains seven
measured rounds, matching V1-01; choose an even count for balanced order. No CPU
pinning or cache flush is used. The separate 29-column comparison CSV emits two
rows per workload, including median/min/max, all samples, ns/dot, fingerprints,
order policy, kernel ISA macros/private options and common environment/build data.
Old benchmark CSV schemas remain unchanged. Debug/sanitizer CTest uses tiny cases.

With diagnostics enabled, the actual target objects save GCC reports at
`build-release/fma-{generic,hardware}-vectorization-Release.txt` and assembly at
`build-release/CMakeFiles/vectorforge_fma_{generic,hardware}_kernel.dir/benchmarks/similarity_kernel.cpp.s`.
Inspect each object's relocated disassembly with `objdump -drwC`; the linked
executable also contains other functions, so a whole-binary `fmaf` search alone
cannot characterize the hardware kernel.

Observed with GCC 13.4.0: generic has scalar loads and a `fmaf` call per feature;
hardware has scalar `vfmadd231ss`, no `fmaf` call, and the same accumulator register
feeding the next iteration. Both reports show zero vectorized loops and identify
control flow in the loop. `vandps` implements the finite check, not packed
floating-point arithmetic. Neither kernel has a packed floating-point reduction;
enabling FMA did not change the sequential dependency chain.

The saved 12-row run measured about 40 to 16 ns/dot at D=32 (61% lower),
119–121 to 105–108 ns/dot at D=128 (11–12% lower), and 693–769 to 680–770 ns/dot
at D=768 (near parity). Input/output fingerprints match the preserved V1-01
baseline. Samples include scheduling outliers and overlapping ranges at D=768;
these are workload-specific observations from one unpinned WSL2 run. The smaller
gain at larger D is consistent with a sequential dependency bottleneck and lower
relative call overhead, but no profiler attribution is claimed. Generic `fmaf`
may itself use runtime-selected hardware; its implementation was not profiled.
CSV, source/build context and raw evidence stay in ignored `build-release/`.
