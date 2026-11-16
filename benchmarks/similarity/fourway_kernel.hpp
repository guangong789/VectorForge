#pragma once

#include <span>
#include <string_view>

namespace vectorforge::benchmark::accumulator {

// Experimental only. Same finite-input/non-overlap preconditions and shape
// checks as the sequential benchmark. Feature j accumulates into lane j%4,
// including tails; all lanes start at +0. Merge is ((a0+a1)+a2)+a3, with a
// float32 rounding and finite check after every FMA and merge addition.
// This changes accumulation order and is NOT the production numerical contract.
void fourway_scores(std::span<const float> query, std::span<const float> embeddings,
                    std::span<float> scores);
std::string_view fourway_isa();

} // namespace vectorforge::benchmark::accumulator
