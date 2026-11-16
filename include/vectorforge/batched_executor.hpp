#pragma once

#include "vectorforge/query_types.hpp"

#include <span>
#include <vector>

namespace vectorforge {

class ColumnBatch;
class EmbeddingBuffer;
class ValidatedDataset;

// Independent batch scan -> AND filter -> bounded scoring -> global heap Top-K.
// batch_size must be positive, including for empty input and K=0. Sizes exceeding
// the dataset are valid. Only actual batch lengths advance the dataset offset.
// Batch-local feature/embedding rows map to record_ids[batch.global_row(local)].
// One heap spans all batches, keeping at most min(K, candidates_seen) records.
// Its root is the worst retained record; only strictly better arrivals replace it.
// Retained records are sorted in place into final output order after scanning.
// Candidate storage grows with actual retained records, never preallocating K:
// O(min(K, matches)) space, plus constant candidate temporaries for heap operations.
// The same vector becomes the result; no additional full-match array is retained.
// A fixed internal window W=32 holds global row indices and scalar score outputs;
// it flushes when full or at batch end, in scan order. It copies no embeddings.
// Total scoring/candidate auxiliary storage is O(W + min(K,matches)), excluding
// input validation. The window's indices/scores use 384 payload bytes on 64-bit hosts.
// Input validation still uses an O(dataset rows) temporary set for duplicate IDs,
// released before scanning. This is separate from the bounded candidate storage.
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

// Trusted overload: immutable stored inputs were validated at dataset admission.
// Only query/predicate/batch-size validation repeats (also at K=0); no O(N) ID set
// or dataset copy occurs per query. Scoring/result storage is O(W+min(K,matches)),
// with constant auxiliary validation/heap storage. FMA and ranking are unchanged.
[[nodiscard]] std::vector<ScoredRecord> execute_batched(
    const ValidatedDataset& dataset, std::span<const float> query,
    std::span<const Predicate> predicates, std::size_t k, std::size_t batch_size);

} // namespace vectorforge
