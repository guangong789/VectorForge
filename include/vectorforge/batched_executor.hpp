#pragma once

#include "vectorforge/query_types.hpp"

#include <span>
#include <vector>

namespace vectorforge {

class ColumnBatch;
class EmbeddingBuffer;

// Independent batch scan -> AND filter -> scalar FMA -> global full sort -> Top-K.
// batch_size must be positive, including for empty input and K=0. Sizes exceeding
// the dataset are valid. Only actual batch lengths advance the dataset offset.
// Batch-local feature/embedding rows map to record_ids[batch.global_row(local)].
// All batches append to one candidate vector; ranking occurs once after scanning.
//
// Borrows inputs synchronously and retains no views. Results own their ID/score
// values. Owners must remain alive and stationary for the call, with the existing
// V0 ownership contracts. No input storage is copied or mutated.
//
// Matches scalar-reference semantics: exact predicate types; all inputs validated
// before filtering, even for K=0 or no matches (including unused float columns,
// filtered-out embeddings, unique IDs, dimensions and row alignment). K=0 then
// returns without scoring. Only qualifying rows compute scores, starting at +0.0f
// and applying std::fma(query[f], embedding[f], score) in increasing feature order.
// Every float32 accumulated score must be finite. Build without fast-math.
// Sorts by score descending, then ID ascending for exact ties, including signed
// zero; returns min(K, matches) records. No epsilon comparison or per-batch Top-K.
//
// Zero batch size and other invalid inputs throw invalid_argument; invalid column
// indices throw out_of_range; non-finite computed scores throw overflow_error.
[[nodiscard]] std::vector<ScoredRecord> execute_batched(
    const ColumnBatch& columns, const EmbeddingBuffer& embeddings,
    std::span<const RecordID> record_ids, std::span<const float> query,
    std::span<const Predicate> predicates, std::size_t k, std::size_t batch_size);

} // namespace vectorforge
