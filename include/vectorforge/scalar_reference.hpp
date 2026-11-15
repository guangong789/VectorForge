#pragma once

#include "vectorforge/query_types.hpp"

#include <span>
#include <vector>

namespace vectorforge {

class ColumnBatch;
class EmbeddingBuffer;
class ValidatedDataset;

// Synchronous scalar oracle: AND filters -> dot products -> full sort -> Top-K.
// Inputs are borrowed only for the call; returned records own their values.
// IDs are arbitrary uint64_t values aligned by dataset row, never row indices.
// Constants must exactly match column types; there are no numeric conversions.
//
// Before filtering (even for K=0, empty input, or no matches), validates row
// alignment, unique IDs, query dimension, all predicates, and finiteness of ALL
// float columns, embeddings and query values. ColumnBatch/EmbeddingBuffer retain
// their existing construction invariants and lifetime requirements.
// K=0 returns after input validation and computes no scores. Otherwise only
// qualifying rows are scored. Starting from +0.0f, each feature performs
// score = std::fma(query[feature], embedding[feature], score) in increasing feature
// order: float32 accumulator, one fused rounding per step, no reassociation.
// Each accumulated score must remain finite. Build without fast-math semantics.
//
// Results sort by score descending, then ID ascending for exact ties (including
// signed zero); returns min(K, matches) records. No epsilon comparison is used.
// Invalid column indices throw out_of_range; other invalid inputs throw
// invalid_argument; non-finite computed scores throw overflow_error.
[[nodiscard]] std::vector<ScoredRecord> execute_scalar_reference(
    const ColumnBatch& columns, const EmbeddingBuffer& embeddings,
    std::span<const RecordID> record_ids, std::span<const float> query,
    std::span<const Predicate> predicates, std::size_t k);

// Trusted overload: stored data/ID invariants were established at admission.
// Validates query dimensions/finiteness and every predicate, including at K=0;
// does not copy dataset storage or rebuild the duplicate-ID set. The scalar
// scoring/full-sort algorithm and computed-score validation remain unchanged.
[[nodiscard]] std::vector<ScoredRecord> execute_scalar_reference(
    const ValidatedDataset& dataset, std::span<const float> query,
    std::span<const Predicate> predicates, std::size_t k);

} // namespace vectorforge
