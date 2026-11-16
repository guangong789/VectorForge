#pragma once

#include <cstddef>
#include <span>

namespace vectorforge {
class EmbeddingBuffer;

namespace detail {

// Fixed query-local window, independent of N/K/user batch size. 32 amortises the
// synchronous call with modest stack storage: 32 size_t indices + 32 float scores
// (384 payload bytes on 64-bit hosts). No embedding payloads are copied.
inline constexpr std::size_t candidate_window_capacity = 32;

// Private synchronous scalar operation; retains no views and allocates nothing
// on valid inputs. Query/storage finiteness was checked by the query entry point.
// Scores input global rows in their supplied order, with one +0 float32 sequential
// FMA chain and a finite check after every step. Output slots correspond to rows.
//
// Returns the number of completed finite scores. A shorter prefix means the next
// candidate produced a nonfinite intermediate; no later candidate is processed.
// The caller consumes that prefix before throwing the existing overflow_error.
// This preserves earlier heap-allocation failure precedence from the fused loop,
// without exposing partial query results or allocating an exception in this stage.
// Output length must equal row count (<=capacity); query dimension must match.
std::size_t score_candidates(const EmbeddingBuffer& embeddings, std::span<const float> query,
                             std::span<const std::size_t> global_rows, std::span<float> output);

} // namespace detail
} // namespace vectorforge
