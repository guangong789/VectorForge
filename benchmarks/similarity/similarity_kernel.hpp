#pragma once

#include <span>
#include <string_view>

namespace vectorforge::benchmark::similarity {

// Benchmark-only kernel. Inputs must be finite and non-overlapping with output.
// Embeddings are contiguous row-major; scores.size() is the row count.
// Includes shape checks, sequential +0 float32 FMA, a finite check after each
// FMA, and one output store per row. No allocation, filtering or ranking.
void sequential_fma_scores(std::span<const float> query, std::span<const float> embeddings,
                           std::span<float> scores);

// Compile-time ISA macros of this kernel TU, not runtime dispatch capabilities.
std::string_view kernel_isa();

} // namespace vectorforge::benchmark::similarity
