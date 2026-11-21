# VectorForge

VectorForge is a C++20 columnar filtered-retrieval engine. Its production library
runs on the CPU with strict numerical semantics. Standalone CUDA experiments
investigate scoring, residency, and the cost of serving filtered queries.

## Production CPU core

- Numeric columns hold `int32_t`, `int64_t`, or `float`. Embeddings are contiguous,
  row-major float32. Read-only batches expose views into both owners.
- `ValidatedDataset` admits an immutable snapshot, checks aligned row counts,
  finite floating-point values, and unique RecordIDs, then reuses that validation
  across queries. Raw query entry points validate their inputs on each call.
- Exact filtered dot-product Top-K has an independent scalar reference and a
  batched executor. The batched path scores bounded candidate windows and
  retains at most K results in a heap. Results sort by descending score and
  ascending RecordID for ties.
- Filtered COUNT, SUM, and fused affine SUM also accept numeric-only inputs.
  SUM and affine SUM use ordered, overflow-checked int64 accumulation.

`ColumnBatch` and `EmbeddingBuffer` own their storage. Keep owners alive and
stationary while using their `BatchView` objects or spans. A moved owner requires
new views. Query results own their RecordIDs and scores.

### Numerical contracts

**Contract S** preserves the CPU executor's observable validation and failure
order, floating-point environment, math provider, and `errno` and exception
flags where applicable. Dot products apply one float32 FMA per dimension in
order. Production candidate scoring uses the generic scalar path.

**Contract G** is an explicit experimental CUDA choice. It operates on finite
float32 inputs, uses sequential round-to-nearest FMA with gradual underflow,
and reports the first non-finite intermediate per candidate slot. Contract G
has a narrower scope for CPU floating-point side effects and error precedence.

## Build and test

Run from the repository root with CMake 3.20+ and a C++20 compiler.

```sh
vf=$(pwd -P)
work=$(mktemp -d)
cmake -S "$vf" -B "$work/cpu" -DCMAKE_BUILD_TYPE=Release
cmake --build "$work/cpu"
ctest --test-dir "$work/cpu" --output-on-failure
```

The root build requires only a C++ toolchain. Consumers link
`VectorForge::vectorforge`. Set `BUILD_TESTING=OFF` to omit the core tests.
A supported GCC or Clang build can enable ASan/UBSan with
`-DVECTORFORGE_ENABLE_SANITIZERS=ON`. [CPU benchmark instructions](benchmarks/README.md)

### Independent integration example

The [CPU retrieval consumer](examples/integration/ai_retrieval/main.cpp) exercises
the library through public headers and its CMake target. Reuse the shell
variables above.

```sh
cmake -S "$vf/examples/integration/ai_retrieval" -B "$work/consumer" \
  -DCMAKE_BUILD_TYPE=Release -DVECTORFORGE_SOURCE_DIR="$vf"
cmake --build "$work/consumer"
ctest --test-dir "$work/consumer" --output-on-failure
```

## Optional CUDA research

The standalone experiments filter metadata on the CPU, send ordered candidate
indices to a private CUDA stream, score against GPU-resident embeddings, and
verify and rank the returned scores on the CPU. The historical H0 automatic
placement prototype is limited to its measured S90 eight-query resident
workload and verified environment. The V2 entry points cover row locality,
fixed-candidate preparation, and mixed-candidate retrieval.

The [CUDA research guide](examples/experimental/cuda_scoring/README.md) lists
independent build options, correctness gates, bounded measurement commands,
and the scope of historical findings. Historical raw archives are separate
from Git. Production GPU serving, dynamic placement, GPU candidate caching,
approximate indexing, and distributed execution are outside the current scope.

## Repository layout

| Path | Contents |
| --- | --- |
| `include/vectorforge/`, `src/` | Public C++ API and CPU implementation |
| `tests/` | Data, numerical, executor, and integration-boundary regressions |
| `examples/integration/ai_retrieval/` | Public API consumer and independent workload oracle |
| `examples/experimental/cuda_scoring/` | Standalone H0 and V2 CUDA experiments |
| `benchmarks/` | Optional CPU performance and numerical studies |
