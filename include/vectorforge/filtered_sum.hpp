#pragma once

#include "vectorforge/query_types.hpp"

#include <span>

namespace vectorforge {

class ColumnBatch;

// Synchronous numeric-only SUM; no embeddings or RecordIDs are required. Borrows
// columns/predicates only for this call; owners and any retained source aliases
// must not mutate/move their storage during it. The returned int64 owns its value.
//
// Validation order, also for zero rows/no matches:
// 1. batch_size > 0;
// 2. all column lengths and all float column values (including unused columns)
//    are checked for row alignment and finiteness, in column/row order;
// 3. target_column exists and has exactly int32 type;
// 4. every predicate is validated, in supplied order, using the existing exact
//    types, six comparisons and finite float constants. No numeric conversion.
// Multiple predicates use short-circuit AND; empty predicates select every row.
//
// Selected int32 values are widened and accumulated into int64 in ascending
// global row order, independently of batch size. Each addition is checked BEFORE
// it occurs. The first overflowing prefix throws overflow_error, even if later
// values could cancel it; there is no wraparound, saturation or partial result.
// No selected rows returns int64 zero. No SQL NULL semantics.
// Invalid target/predicate column index: out_of_range. Zero batch size, malformed
// predicates, type mismatch, nonfinite float data/constants or misaligned columns:
// invalid_argument. A moved-from ColumnBatch is empty, so has no valid target.
//
// O(1) auxiliary space and no successful-path heap allocation/copies. Validation
// scans all float data; execution is O(rows * (1 + predicates)) in the worst case.
// Batch size bounds traversal ranges; no matching-row array is materialized.
[[nodiscard]] std::int64_t sum_where(const ColumnBatch& columns, std::size_t target_column,
                                   std::span<const Predicate> predicates, std::size_t batch_size);

} // namespace vectorforge
