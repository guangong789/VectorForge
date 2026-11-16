#pragma once

#include "candidate_scoring.hpp"

#include <limits>

namespace vectorforge::detail {

// Private integration/validation surface, never included by public headers.
// The original generic scalar kernel is authoritative in all FP environments.
std::size_t score_candidates_scalar(const EmbeddingBuffer&, std::span<const float>,
                                    std::span<const std::size_t>, std::span<float>);

// Nonthrowing eligibility check; an ineligible address must use the scalar path,
// not introduce an exception ahead of an earlier candidate's scalar failure.
inline bool simd_row_addressable(std::size_t row, std::size_t dimension) noexcept {
    constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float);
    return dimension != 0 && dimension - 1 <= limit && row <= (limit - (dimension - 1)) / dimension;
}

bool simd_hardware_supported() noexcept;
bool simd_platform_supported() noexcept;
bool simd_environment_supported() noexcept;
bool simd_scoring_eligible(const EmbeddingBuffer&, std::span<const float>,
                           std::span<const std::size_t>, std::span<float>) noexcept;

// Explicit internal experiment; provider equivalence is not established.
// Controls are checked afresh on every call.
// Each full group checkpoints its incoming environment/errno; on failure only
// that group is restored and replayed by the generic scalar implementation.
// Production score_candidates unconditionally uses the generic scalar scorer;
// only dedicated tests/benchmarks call this candidate directly.
std::size_t score_candidates_auto(const EmbeddingBuffer&, std::span<const float>,
                                  std::span<const std::size_t>, std::span<float>);

// Native non-LTO object: exactly eight valid rows after generic preflight. False
// reports an intermediate nonfinite lane, with no output writes or scalar replay.
bool score_candidates_eight_avx2(const EmbeddingBuffer&, std::span<const float>,
                                 std::span<const std::size_t>, std::span<float>);
} // namespace vectorforge::detail
