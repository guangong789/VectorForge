# VectorForge

A minimal C++20 data foundation for numeric columns and dense float32 embeddings.
V0-01 provides storage and read-only batches; execution operators are outside this milestone.
There are no dependencies beyond the C++ standard library and CMake (3.20 or newer).

## Build and test

With a C++20 compiler (GCC, Clang, or MSVC):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

Optional ASan/UBSan build with a supported GCC/Clang toolchain:

```sh
cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug -DVECTORFORGE_ENABLE_SANITIZERS=ON
cmake --build build-sanitize
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 ctest --test-dir build-sanitize --output-on-failure
```

If LeakSanitizer reports that it cannot run under ptrace in a sandbox/debugger,
use `ASAN_OPTIONS=detect_leaks=0` for that command. This disables leak detection
only; AddressSanitizer and UBSan remain enabled.

Tests remain active in release builds; they do not rely on `assert`. `BUILD_TESTING=OFF`
builds only the core library. Consumers link `VectorForge::vectorforge`.

## Data and row semantics

`ColumnBatch` owns a vector of columns, each a variant of `vector<int32_t>`,
`vector<int64_t>`, or `vector<float>`. All columns must have equal lengths. Row count
is inferred from the first column. No columns means zero rows; representing positive
row counts without any numeric columns is outside this minimal interface. Empty
typed columns are allowed. There are no NULL values or implicit type conversions.
`column<T>(index)` returns a `span<const T>`; unsupported template types fail to compile.

`EmbeddingBuffer(rows, dimension, values)` owns one contiguous vector of IEEE 754
float32 values. Dimension must be positive, even for zero rows. The size must equal
`rows * dimension`; multiplication is checked for `size_t` overflow before use.
`row(index)` returns a `span<const float>` with exactly `dimension` elements. The
element at dataset row `r`, feature `f` is stored at `r * dimension + f`. There are
no per-row allocations. Row bounds are checked before offset multiplication.

`BatchView(columns, embeddings, offset, count)` borrows an exact half-open row range.
It requires equal total row counts for both owners and checks the range using
subtraction so large invalid arguments cannot wrap around. `make_batch` accepts a
positive batch size and clamps the count to the remaining rows. At the dataset end
it returns an empty view; offsets beyond the end are rejected. Empty ranges inside
the dataset are valid. No padding or data copying occurs.

`row_offset()` is dataset-global. Column span indices and `embedding_row(local_row)`
are batch-local; `global_row(local_row)` performs the checked conversion. Column
indices always refer to the original column order. A zero batch size is invalid.
For example:

```cpp
#include <vectorforge/batch_view.hpp>
#include <cstdint>
#include <utility>
#include <vector>

std::vector<vectorforge::ColumnBatch::Column> storage;
storage.emplace_back(std::vector<std::int32_t>{10, 20, 30});
const vectorforge::ColumnBatch columns(std::move(storage));
const vectorforge::EmbeddingBuffer embeddings(3, 2, {1, 2, 3, 4, 5, 6});

const std::size_t batch_size = 2;
for (std::size_t offset = 0; offset < columns.row_count();) {
    const auto batch = vectorforge::make_batch(columns, embeddings, offset, batch_size);
    const auto numbers = batch.column<std::int32_t>(0);
    for (std::size_t local = 0; local < batch.row_count(); ++local) {
        const auto global = batch.global_row(local);
        const auto embedding = batch.embedding_row(local);
        // Use numbers[local], embedding[feature], and global here.
        (void)numbers;
        (void)global;
        (void)embedding;
    }
    offset += batch.row_count(); // Advance by actual count, including the tail.
}
```

Empty input executes zero loop iterations; a direct `make_batch(..., 0, batch_size)`
still returns a valid empty view.

Invalid lengths, dimensions, stored-type access, row alignment, and zero batch sizes
throw `std::invalid_argument`. Invalid column indices, row indices, and batch ranges
throw `std::out_of_range`. Shape multiplication overflow throws `std::overflow_error`.
Indexing a returned span with `operator[]` is **not checked**: callers must keep
feature indices below `dimension()` and column-span indices below `row_count()`.

## Ownership and lifetime contract

- `ColumnBatch` and `EmbeddingBuffer` own their vectors through RAII. Pass vector
  storage with `std::move` to transfer allocations. Passing an lvalue vector copies
  explicitly by value; initializer lists may also copy their elements.
- Both owners are noncopyable and expose no mutation, resizing, replacement, or
  assignment operations. Move construction is supported. Moved-from columns become
  a zero-column, zero-row batch; moved-from embeddings become zero-row storage with
  the original dimension. Both remain usable as empty owners.
- `BatchView` and every returned `std::span` are non-owning. Copying a view is cheap
  and never extends either owner's lifetime. A span may outlive the `BatchView`
  that produced it if its backing owner remains alive and stationary.
- The caller must keep both backing owners alive and must discard all their batch
  views and spans before moving or destroying either owner. After a move, create
  new views against the destination. Owner addresses matter because `BatchView`
  stores pointers to the owner objects. Moving an enclosing object/container can
  also move an owner and invalidate this contract.
- Borrowing directly from temporary owners is rejected by deleted rvalue overloads.
  This catches common mistakes, but C++ and `std::span` do not generally prevent
  dangling references. Callers remain responsible for lifetimes, including owners
  in smaller scopes or containers. Do not mutate storage through retained mutable
  aliases after transferring it to an owner.

## Tests

A small standard-library test executable checks valid and invalid inputs, all three
numeric types, read-only API contracts at compile time, storage transfer and shared
view pointers, moved-from invariants, zero/single-row inputs, exact and partial
partitions, row alignment, and arithmetic overflow without huge allocations.
