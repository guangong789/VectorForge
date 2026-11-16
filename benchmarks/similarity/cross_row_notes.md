# Cross-row AVX2/FMA feasibility prototype (V1-07A)

Opt-in experiment; production sources, public API and W=32 are unchanged.
The generic driver checks AVX2/FMA before calling either specialized object.
Both objects use identical private `-march=x86-64 -mtune=generic -mavx2 -mfma
-fno-fast-math` options. Neither enables reassociation or intra-record reduction.

```sh
cmake -S . -B /tmp/vectorforge-cross-row-release -DCMAKE_BUILD_TYPE=Release \
  -DVECTORFORGE_BUILD_BENCHMARKS=ON -DVECTORFORGE_BUILD_CROSS_ROW_STUDY=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build /tmp/vectorforge-cross-row-release -j 2 \
  --target vectorforge_cross_row_tests vectorforge_cross_row_benchmark
ctest --test-dir /tmp/vectorforge-cross-row-release -R '^cross_row_' --output-on-failure
/tmp/vectorforge-cross-row-release/vectorforge_cross_row_benchmark \
  > /tmp/vectorforge-cross-row-release/cross-row.csv
objdump -drwC /tmp/vectorforge-cross-row-release/CMakeFiles/vectorforge_cross_row_avx2.dir/benchmarks/similarity/cross_row_avx2.cpp.o \
  > /tmp/vectorforge-cross-row-release/cross-row.asm
```

Configure a second `/tmp` directory with Debug and
`-DVECTORFORGE_ENABLE_SANITIZERS=ON`, build only `vectorforge_cross_row_tests`,
then run CTest with `-R '^cross_row_numerics'`. Use the established ordinary-user
path outside the restricted sandbox with `ASAN_OPTIONS=detect_leaks=1` and
`UBSAN_OPTIONS=halt_on_error=1`. Hardware gather accesses are not assumed to be
individually instrumented by ASan: explicit prevalidation establishes each lane's
bounds, and focused tests check the last row, invalid lanes/tails and arithmetic
limits without allocating enormous datasets.

Each 8-row group uses two `vgatherqps` operations (four signed 64-bit element
offsets each, scale=4) per feature, combined for one packed float32 FMA. Offsets
cover the entire selected row and are checked against `PTRDIFF_MAX/sizeof(float)`
before multiplication. Only complete 8-row groups gather; 1–7 tails are scalar.
Empty selections never access row zero. There is no transpose, embedding copy or
candidate array proportional to the dataset. Per-call temporaries are eight
64-bit offsets and fixed registers, plus scalar fallback state; at most 32 rows
are accepted. Global row indices and output slots are distinct from Record IDs.

Every dimension starts from the prior lane accumulator and has a finite check.
If a lane becomes nonfinite, replay the bounded group in scalar candidate order
to return the same completed prefix as production, leaving failing/later slots
unwritten. This handles an early row failing in a later dimension than another
lane. Packed evaluation can speculatively affect sticky exception flags before
the replay; floating-point traps/flag/errno timing is outside this experiment's
contract, as it is outside the established scoring API's documented guarantees.

The supported environment is nearest rounding, FTZ/DAZ disabled and masked FP
exceptions. Driver/tests inspect MXCSR and rounding; they skip other environments
rather than silently changing production assumptions. Independent generic
`std::fma` and the production scorer are exact oracles, including subnormals,
signed zeros, cancellation, close scores and intermediate overflow.

The fixed default matrix has 24 cases: two nominal row counts (256/8192), three
dimensions (32/128/768), group sizes 7/32, contiguous/sparse selections. Sixteen
precomputed windows and result arrays are bounded independently of N. The sparse
schedule uses `(ordinal*263+17)%N`; dense uses `ordinal%N`. The odd stride is
coprime to both row counts and is nonconsecutive even modulo 256. Selected working sets
are **not** a full scan of nominal storage: CSV records unique selected rows as
well as embedding bytes. Each sample repeats the schedule 16 times, so this is
a warm-cache candidate-window study, not E2E or cold-cache performance evidence.
Two warm-ups and eight measured samples per variant alternate first position
exactly evenly. All shape/index checks, gather/load, FMA, per-step finite checks
and stores are timed; generation, allocations, verification and output are not.
Result bits are checked after every sample. External calls with IPO disabled and
observable result checks prevent dead-code elimination. Seed, raw samples,
median/min/max, throughput, flags, compiler/CPU/OS and FP environment enter CSV.
Unpinned WSL2 measurements can show noise and regressions; no universal speedup
or production-integration claim follows from a positive feasibility result.
