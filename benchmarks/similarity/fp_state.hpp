#pragma once

#include "vectorforge/embedding_buffer.hpp"

#include <span>

namespace vectorforge::benchmark::fp_state {
using Kernel = std::size_t (*)(const EmbeddingBuffer&, std::span<const float>,
                              std::span<const std::size_t>, std::span<float>);

// Study only. Same admitted-finite inputs and eligibility as the private SIMD
// candidate; unsupported states/shapes/addresses use the original generic scorer.
// Successful groups keep their flags. A failed group's checkpoint (including
// incoming flags and earlier successful groups) is restored before scalar replay.
// No production caller references this function.
std::size_t isolated_scores(const EmbeddingBuffer&, std::span<const float>,
                             std::span<const std::size_t>, std::span<float>);

// Historical V1-07B negative control retained only inside this study. Production
// dispatch no longer exposes an unisolated path after V1-07D integration.
std::size_t unisolated_scores(const EmbeddingBuffer&, std::span<const float>,
                               std::span<const std::size_t>, std::span<float>);

// ISA-isolated, no replay/state handling: exact eight-row group, fully preflighted
// by isolated_scores. False means an intermediate nonfinite lane; no stores occur.
bool eight_scores(const EmbeddingBuffer&, std::span<const float>,
                   std::span<const std::size_t>, std::span<float>);

// Generic-compiled external step for causal localization; not constant-folded
// together with the fixture generator. Inspected for an actual fmaf call.
float scalar_step(float query, float value, float previous);
} // namespace vectorforge::benchmark::fp_state
