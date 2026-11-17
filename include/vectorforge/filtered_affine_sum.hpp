#pragma once

#include "vectorforge/query_types.hpp"

#include <span>

namespace vectorforge {

class ColumnBatch;

// Numeric-only fused AND filter -> affine map -> ordered SUM. Borrows columns and
// predicates only for this call; owners and retained source aliases must not move
// or mutate storage during it. Returns an owning int64, retaining no views.
// No embeddings, RecordIDs, transformed column or selection array are required.
//
// Validation order, also for empty input/no matches/a=0/b=0:
// 1. batch_size > 0;
// 2. each column's length, then its float finiteness, in column order (including
//    unused float columns), exactly as the numeric-only SUM admission model;
// 3. source_column exists and has exactly int32 type;
// 4. all predicates, in supplied order: exact types, six comparisons, finite
//    float constants. Empty predicates match all rows; matching short-circuits AND.
//
// Only selected rows compute y = int64(a) * int64(x) + int64(b), multiplying first.
// The full int32 domain is valid for x/a/b. The product fits int64, and y lies in
// [-2^62, 2^62+2^31-1], so neither operation needs a transform-overflow branch.
// Never compute in int32 first, narrow y, truncate, saturate or restrict coefficients.
//
// Start from int64 zero and consume y in ascending global row order, independently
// of batch size. Check BEFORE each int64 addition. The first overflowing prefix
// throws overflow_error; later cancellation cannot rescue it. Do not reassociate
// or replace this with a*SUM(x)+b*COUNT. No selected rows returns zero; no NULLs or
// partial result. Invalid column index: out_of_range. Zero batch size, misaligned
// columns, nonfinite inputs/constants, wrong types or malformed predicates:
// invalid_argument. A moved-from ColumnBatch is empty and has no valid source.
//
// Single-threaded, O(1) auxiliary space and no successful-path heap allocations
// or payload copies. Validation scans float data; execution is O(rows*(1+predicates))
// in the worst case. Batches bound traversal ranges; inputs are not snapshotted.
[[nodiscard]] std::int64_t sum_affine_where(
    const ColumnBatch& columns, std::size_t source_column, std::int32_t a, std::int32_t b,
    std::span<const Predicate> predicates, std::size_t batch_size);

} // namespace vectorforge
