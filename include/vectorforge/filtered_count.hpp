#pragma once

#include "vectorforge/query_types.hpp"

#include <span>

namespace vectorforge {

class ColumnBatch;
class ValidatedDataset;

// Synchronous filtered row count on an admitted, immutable dataset. Borrowed
// inputs need only remain alive for this call; nothing is retained or mutated.
// Predicates use the existing exact types, six comparisons and AND semantics.
// Validates every predicate and batch_size>0 before processing, even with no
// rows/no matches. Empty predicates match all rows (metadata-only fast path).
// No query vector or K; no scoring, embedding-payload reads, heap, dataset copy,
// ID set or selection array. Storage still follows ValidatedDataset's existing
// embedding-alignment/admission requirements.
//
// Returns size_t because row_count() is size_t, and 0 <= result <= row_count();
// each row is counted at most once, so accumulation cannot overflow size_t.
// O(1) auxiliary space; O(predicates) validation plus O(rows*predicates) scan in
// the worst case. No allocation on successful query paths.
// Invalid column index: out_of_range. Invalid comparison/type/nonfinite float
// constant or zero batch size: invalid_argument. Moved-from dataset: logic_error.
[[nodiscard]] std::size_t count_where(const ValidatedDataset& dataset,
                                    std::span<const Predicate> predicates, std::size_t batch_size);

// Numeric-only raw-input overload: no embeddings, RecordIDs or admission needed.
// Borrows columns/predicates synchronously; retains nothing, copies no payloads.
// Owners and retained source aliases must not mutate/move storage during the call.
// Returns an owning size_t value in [0, row_count()], so counting cannot overflow.
// Each matching row is counted once; zero rows/no matches return zero. No NULLs.
//
// Validation order, even for empty input/empty predicates:
// 1. batch_size > 0;
// 2. ALL column lengths equal row_count(), in column order;
// 3. ALL float values are finite (including unused columns), in column/row order;
// 4. ALL predicates are valid, in supplied order, with the existing exact types,
//    six comparisons and finite float constants. Multiple predicates use
//    short-circuit AND; empty predicates then return row_count() without a scan.
// Otherwise scans bounded ranges in ascending global row order, independently of
// batch size. A moved-from ColumnBatch is empty; predicates still get validated.
//
// Invalid predicate column index: out_of_range. Zero batch size, malformed
// predicates, type mismatch, misaligned columns or nonfinite float data/constants:
// invalid_argument. No partial result on failure. O(1) auxiliary memory, no
// successful-path heap allocations or selection array. Validation scans all float
// data; matching is O(rows * predicates) in the worst case. Trusted overload above
// retains its admission guarantees and does not perform these raw-input scans.
[[nodiscard]] std::size_t count_where(const ColumnBatch& columns,
                                    std::span<const Predicate> predicates, std::size_t batch_size);

} // namespace vectorforge
