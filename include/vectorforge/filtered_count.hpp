#pragma once

#include "vectorforge/query_types.hpp"

#include <span>

namespace vectorforge {

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

} // namespace vectorforge
