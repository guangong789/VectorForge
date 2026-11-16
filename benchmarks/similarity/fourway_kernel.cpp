#include "fourway_kernel.hpp"

#include <cmath>
#include <stdexcept>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "The accumulator experiment requires no-fast-math"
#endif

namespace vectorforge::benchmark::accumulator {
namespace {

float checked(float value) {
    if (!std::isfinite(value)) { throw std::overflow_error("four-way score is not finite"); }
    return value;
}

} // namespace

void fourway_scores(std::span<const float> query, std::span<const float> embeddings,
                    std::span<float> scores) {
    if (query.empty() || embeddings.size() % query.size() != 0 ||
        embeddings.size() / query.size() != scores.size()) {
        throw std::invalid_argument("require D>0 and exactly rows*D embedding values");
    }
    for (std::size_t row = 0; row < scores.size(); ++row) {
        const auto* values = embeddings.data() + row * query.size();
        float a0 = 0.0F, a1 = 0.0F, a2 = 0.0F, a3 = 0.0F;
        std::size_t j = 0;
        for (; query.size() - j >= 4; j += 4) {
            a0 = checked(std::fma(query[j], values[j], a0));
            a1 = checked(std::fma(query[j + 1], values[j + 1], a1));
            a2 = checked(std::fma(query[j + 2], values[j + 2], a2));
            a3 = checked(std::fma(query[j + 3], values[j + 3], a3));
        }
        if (j < query.size()) { a0 = checked(std::fma(query[j], values[j], a0)); ++j; }
        if (j < query.size()) { a1 = checked(std::fma(query[j], values[j], a1)); ++j; }
        if (j < query.size()) { a2 = checked(std::fma(query[j], values[j], a2)); }
        float score = checked(a0 + a1);
        score = checked(score + a2);
        scores[row] = checked(score + a3);
    }
}

std::string_view fourway_isa() {
    return "x86_64"
#if defined(__SSE2__)
        ";SSE2"
#endif
#if defined(__AVX__)
        ";AVX"
#endif
#if defined(__AVX2__)
        ";AVX2"
#endif
#if defined(__AVX512F__)
        ";AVX512F"
#endif
#if defined(__FMA__)
        ";FMA"
#endif
        ;
}

} // namespace vectorforge::benchmark::accumulator
