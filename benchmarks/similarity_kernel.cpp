#include "similarity_kernel.hpp"

#include <cmath>
#include <stdexcept>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0) || defined(_M_FP_FAST)
#error "The similarity baseline requires sequential FMA without fast-math"
#endif

// Compile the identical benchmark algorithm into isolated target namespaces.
// Production code and the original V1-01 symbol do not depend on this macro.
#ifndef VECTORFORGE_SIMILARITY_NAMESPACE
#define VECTORFORGE_SIMILARITY_NAMESPACE vectorforge::benchmark::similarity
#endif
namespace VECTORFORGE_SIMILARITY_NAMESPACE {

void sequential_fma_scores(std::span<const float> query, std::span<const float> embeddings,
                           std::span<float> scores) {
    if (query.empty() || embeddings.size() % query.size() != 0 ||
        embeddings.size() / query.size() != scores.size()) {
        throw std::invalid_argument("require D>0 and exactly rows*D embedding values");
    }
    for (std::size_t row = 0; row < scores.size(); ++row) {
        float score = 0.0F;
        for (std::size_t feature = 0; feature < query.size(); ++feature) {
            score = std::fma(query[feature], embeddings[row * query.size() + feature], score);
            if (!std::isfinite(score)) {
                throw std::overflow_error("similarity score is not finite");
            }
        }
        scores[row] = score;
    }
}

std::string_view kernel_isa() {
    return
#if defined(__x86_64__) || defined(_M_X64)
        "x86_64"
#elif defined(__aarch64__)
        "aarch64"
#else
        "other_arch"
#endif
#if defined(__SSE2__) || defined(_M_X64)
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
#if defined(__ARM_NEON)
        ";NEON"
#endif
        ;
}

} // namespace vectorforge::benchmark::similarity
