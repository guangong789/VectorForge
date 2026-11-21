[简体中文](README.md) | English

# VectorForge

**A C++20 engine for exact filtered vector retrieval**

VectorForge filters in-memory records on typed columnar metadata and computes exact dot-product Top-K over contiguous row-major FP32 embeddings. The project focuses on C++ data ownership, numerical correctness, and performance engineering. The production library runs on the CPU. Optional standalone CUDA experiments study GPU scoring and data residency.

## Architecture

~~~mermaid
flowchart LR
    input["Query and data"]
    subgraph cpu_path["Production CPU · Contract S"]
        cpu_filter["CPU predicate filtering"] --> cpu_score["CPU scalar scoring"] --> cpu_topk["CPU bounded-heap Top-K"]
    end
    subgraph cuda_path["CUDA experiment · Contract G"]
        gpu_filter["CPU predicate filtering"] --> gpu_score["GPU scoring (resident embeddings)"] --> gpu_topk["CPU validation and Top-K"]
    end
    input --> cpu_filter
    input --> gpu_filter
~~~

## Core C++ design

- **Ownership and views.** `ColumnBatch` and `EmbeddingBuffer` manage storage through RAII. `BatchView` and read-only `std::span` borrow that storage, so slicing a batch requires no copy. Query results own their RecordIDs and scores.
- **Immutable snapshots.** `ValidatedDataset` deep-copies the data at admission and checks row alignment, floating-point finiteness, and RecordID uniqueness. Trusted queries reuse those checks while still validating each query vector and its predicates.
- **Bounded ranking.** The batched executor filters rows in batches and buffers up to 32 candidate row indices and their scores at a time. A global heap allocates on demand and retains at most `min(K, M)` entries, where `M` is the number of matches. The candidate window and ranking use `O(32 + min(K, M))` additional storage. Dataset storage, admission allocations, and raw-input validation costs are accounted for separately. The independent scalar reference collects and sorts all matches for differential testing. Both paths order results by descending score and break ties by ascending RecordID.
- **Streaming aggregation.** COUNT, SUM, and fused affine SUM operate directly on numeric columns, with no dynamic allocation on success. SUM and affine SUM take an `int32_t` column, accumulate into `int64_t` in row order, and check every addition for overflow.

## Numerical semantics and performance engineering

- **CPU Contract S.** Dot products start at `+0.0f`, apply FP32 FMA in dimension order, and check every intermediate for finiteness. Compilation requires fast-math to be disabled. The contract also preserves observable validation and failure order, as well as the applicable floating-point environment, math provider behavior, exception flags, and `errno`. Differential tests compare score bits and result order.
- **CUDA Contract G.** The experimental path uses a separate contract. For finite FP32 inputs, it uses round-to-nearest FMA in dimension order with gradual underflow and records the dimension of the first non-finite intermediate for each candidate. The CPU checks the returned status before publishing Top-K. [CUDA research guide](examples/experimental/cuda_scoring/README.md)
- **Trusted CPU query comparison.** In one WSL2 Release comparison with `N=8192`, `D=128`, 100% selectivity, `K=10`, and batch size 256, trusted query time went from **1.93 ms to 1.00 ms** after introducing a separate scorer and a fixed candidate window. These are medians of run medians across six paired runs. Timing covers the full trusted query, with admission measured separately. The gain depends on the entry point and workload. [CPU benchmarks and reproduction steps](benchmarks/README.md#bounded-production-candidate-scoring-v1-06)

## Quick start

Run from the repository root. The CPU build requires only CMake 3.20 or later and a C++20 compiler.

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
~~~

## Design decisions and boundaries

- **Layout follows access patterns.** Columnar metadata lets filtering scan the columns it needs. Row-major embeddings keep each record's dimensions contiguous for scoring.
- **Admission pays for reusable validation.** Immutable snapshots are isolated from aliases to the original storage. Repeated queries avoid full dataset validation and duplicate-ID checks. The time saved per query and the reuse count determine whether the initial copy pays off. [Admission and reuse benchmark](benchmarks/README.md#validated-admission-and-query-reuse-v0-08)
- **Ranking memory stays bounded as matches grow.** With small K, a bounded heap avoids materializing all matches. As K approaches the match count, the space advantage shrinks and heap maintenance still has a cost. [Isolated Top-K comparison](benchmarks/README.md#isolated-top-k-comparison-v0-06)
- **Production scoring always uses the generic scalar path.** Cross-row SIMD can preserve each record's FMA order, but a known counterexample changes `errno` even when score bits and floating-point exception flags agree. SIMD remains in standalone experiments. [SIMD validation notes](src/scoring/simd_integration_notes.md)
- **Execution scope.** The CPU core performs single-host, in-memory exact scans, ranking, and aggregation. Standalone CUDA experiments study resident scoring, data transfer, and query reuse. The automatic placement prototype is limited to its verified historical workload and environment.

## Repository layout

| Path | Contents |
| --- | --- |
| [include/vectorforge/](include/vectorforge/), [src/](src/) | Public C++ API and CPU implementation |
| [tests/](tests/) | Data, executor, and numerical contract tests |
| [benchmarks/](benchmarks/) | CPU query, ranking, and scoring experiments |
| [examples/integration/ai_retrieval/](examples/integration/ai_retrieval/) | Independent C++ consumer and retrieval correctness checks |
| [examples/experimental/cuda_scoring/](examples/experimental/cuda_scoring/) | Standalone CUDA scoring, residency, and placement experiments |
