#include "similarity/cross_row.hpp"
#include "scoring/candidate_scoring.hpp"
#include "vectorforge/query_types.hpp"

#include <array>
#include <bit>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {
using namespace vectorforge;
namespace study = benchmark::cross_row;
std::size_t comparisons = 0;
void check(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}
template<class E, class F> void throws(F f) {
    try { f(); } catch (const E&) { return; }
    throw std::runtime_error("expected exception missing");
}

// Independent generic-compiled oracle: no experimental arithmetic/helper reuse.
std::size_t oracle(const EmbeddingBuffer& buffer, std::span<const float> query,
                   std::span<const std::size_t> rows, std::span<float> scores) {
    for (std::size_t i = 0; i < rows.size(); ++i) {
        float score = 0.0F;
        for (std::size_t d = 0; d < query.size(); ++d) {
            score = std::fma(query[d], buffer.row(rows[i])[d], score);
            if (!std::isfinite(score)) { return i; }
        }
        scores[i] = score;
    }
    return rows.size();
}

std::size_t compare(const EmbeddingBuffer& buffer, std::span<const float> query,
                    std::span<const std::size_t> rows) {
    std::array<float, study::capacity + 2> expected;
    expected.fill(-12345.25F);
    const auto completed = oracle(buffer, query, rows, std::span(expected).subspan(1, rows.size()));
    for (const auto kernel : {study::scalar_scores, study::avx2_scores, detail::score_candidates}) {
        for (int repeat = 0; repeat < 3; ++repeat) {
            std::array<float, study::capacity + 2> actual;
            actual.fill(-12345.25F);
            const auto result = kernel(buffer, query, rows, std::span(actual).subspan(1, rows.size()));
            check(result == completed, "finite prefix must agree with independent oracle");
            for (std::size_t i = 0; i < expected.size(); ++i) {
                check(std::bit_cast<std::uint32_t>(actual[i]) == std::bit_cast<std::uint32_t>(expected[i]),
                      "score bits, row mapping, unwritten suffix or output canary mismatch");
            }
            ++comparisons;
        }
    }
    return completed;
}

void dimensions_and_groups() {
    std::mt19937 random(0x56463107U);
    auto value = [&] {
        const auto mantissa = static_cast<int>(random() % 2001) - 1000;
        return std::ldexp(static_cast<float>(mantissa) / 997.0F, static_cast<int>(random() % 13) - 8);
    };
    for (const auto dimension : {1U, 2U, 3U, 7U, 8U, 9U, 31U, 32U, 33U, 128U, 129U, 768U}) {
        std::vector<float> data(97 * dimension), query(dimension);
        std::generate(data.begin(), data.end(), value);
        std::generate(query.begin(), query.end(), value);
        const EmbeddingBuffer buffer(97, dimension, std::move(data));
        for (const auto count : {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 15U, 16U, 17U, 31U, 32U}) {
            std::array<std::size_t, study::capacity> indices{};
            for (int mode = 0; mode < 3; ++mode) {
                for (std::size_t i = 0; i < count; ++i) {
                    indices[i] = mode == 0 ? 97 - count + i : mode == 1 ? (i * 37 + 13) % 97 : 96;
                }
                check(compare(buffer, query, std::span(indices).first(count)) == count, "finite data must complete");
            }
        }
    }
    const EmbeddingBuffer empty(0, 3, {});
    const std::array query{1.0F, 2.0F, 3.0F};
    compare(empty, query, {});
}

void numerical_cases() {
    const float a = std::ldexp(1.0F, -23), tiny = std::numeric_limits<float>::denorm_min();
    const std::array query{-1.0F, 1.0F + a, 1.0F, -0.0F, 1.0F};
    std::vector<float> data(9 * query.size(), 0.0F);
    for (std::size_t row = 0; row < 9; ++row) {
        data[row * 5] = 1;
        data[row * 5 + 1] = 1 - a;
        data[row * 5 + 2] = row % 2 == 0 ? tiny : -tiny;
        data[row * 5 + 3] = -0.0F;
    }
    const EmbeddingBuffer cancellation(9, 5, std::move(data));
    const std::array<std::size_t, 9> rows{8, 0, 6, 3, 1, 7, 4, 2, 5};
    compare(cancellation, query, rows);
    const EmbeddingBuffer subnormal(9, 1, {tiny, -tiny, 0.0F, -0.0F, tiny, -tiny, -0.0F, 0.0F, tiny});
    for (const auto q : {1.0F, -1.0F, 0.0F, -0.0F, 0.5F}) {
        const std::array one{q};
        compare(subnormal, one, rows);
    }
    // Later lane fails at d=0, earlier lane only at d=2; prefix must be row-order.
    const float largest = std::numeric_limits<float>::max();
    std::vector<float> values(9 * 3, 0);
    values[1 * 3 + 2] = largest;
    values[6 * 3] = largest;
    const EmbeddingBuffer overflow(9, 3, std::move(values));
    const std::array q{2.0F, 1.0F, 2.0F};
    const std::array<std::size_t, 9> order{0, 1, 2, 3, 4, 5, 6, 7, 8};
    check(compare(overflow, q, order) == 1, "earliest row wins over earliest SIMD dimension failure");
    const std::array<std::size_t, 9> reversed{0, 6, 2, 3, 4, 5, 1, 7, 8};
    check(compare(overflow, q, reversed) == 1, "first-dimension overflow");
    // Overflow followed by cancellation must fail before a later finite result.
    std::vector<float> recover(9 * 3, 0);
    recover[8 * 3] = largest;
    recover[8 * 3 + 1] = largest;
    recover[8 * 3 + 2] = -largest;
    const EmbeddingBuffer transient(9, 3, std::move(recover));
    const std::array ones{1.0F, 1.0F, 1.0F};
    check(compare(transient, ones, order) == 8, "tail intermediate overflow");
    for (std::size_t failed = 0; failed < 9; ++failed) {
        std::vector<float> v(9, 1);
        v[failed] = largest;
        const EmbeddingBuffer b(9, 1, std::move(v));
        const std::array twice{2.0F};
        check(compare(b, twice, order) == failed, "every lane and scalar tail overflow");
    }
}

void bounds() {
    const EmbeddingBuffer buffer(1, 1, {1});
    const std::array query{1.0F};
    const std::array<std::size_t, 1> good{0};
    for (const auto kernel : {study::scalar_scores, study::avx2_scores}) {
        std::array<float, 1> output{-99};
        throws<std::invalid_argument>([&] { (void)kernel(buffer, {}, good, output); });
        throws<std::invalid_argument>([&] { (void)kernel(buffer, query, good, {}); });
        std::array<std::size_t, 33> oversized{};
        std::array<float, 33> large_output{};
        throws<std::invalid_argument>([&] { (void)kernel(buffer, query, oversized, large_output); });
        for (const auto index : {std::size_t{1}, std::numeric_limits<std::size_t>::max()}) {
            const std::array bad{index};
            throws<std::out_of_range>([&] { (void)kernel(buffer, query, bad, output); });
            check(output[0] == -99, "validate all rows before stores/gather");
        }
        const std::array<std::size_t, 9> later_bad{0, 0, 0, 0, 0, 0, 0, 0, 1};
        std::array<float, 9> untouched;
        untouched.fill(-99);
        throws<std::out_of_range>([&] { (void)kernel(buffer, query, later_bad, untouched); });
        check(std::all_of(untouched.begin(), untouched.end(), [](float x) { return x == -99; }), "invalid tail cannot trigger full-group gather");
    }
    constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float);
    check(study::checked_offset(1ULL << 31, 1) == (1LL << 31), "offsets beyond signed i32 stay intact");
    check(study::checked_offset(limit, 1) == static_cast<std::int64_t>(limit), "maximum byte-distance bound");
    throws<std::overflow_error>([&] { (void)study::checked_offset(limit + 1, 1); });
    throws<std::overflow_error>([&] { (void)study::checked_offset(limit, 2); });
    throws<std::overflow_error>([&] { (void)study::checked_offset(0, std::numeric_limits<std::size_t>::max()); });
    throws<std::invalid_argument>([&] { (void)study::checked_offset(0, 0); });
}

void ranking() {
    const float half_ulp = std::ldexp(1.0F, -24);
    std::vector<float> data(9 * 8, 0);
    for (std::size_t row = 0; row < 9; ++row) { data[row * 8] = 1; }
    data[0] = std::nextafter(1.0F, 2.0F);
    data[8 + 1] = half_ulp;
    data[8 + 5] = half_ulp;
    const EmbeddingBuffer buffer(9, 8, std::move(data));
    const std::array query{1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
    const std::array<std::size_t, 9> rows{8, 1, 6, 2, 4, 7, 5, 3, 0};
    const std::array<RecordID, 9> ids{900, 2, 70, 4, 50, 6, 30, 8, 100};
    std::array<float, 9> a{}, b{};
    check(oracle(buffer, query, rows, a) == 9 && study::avx2_scores(buffer, query, rows, b) == 9, "rank scoring completes");
    auto sorted = [&](const auto& scores) {
        std::array<ScoredRecord, 9> result;
        for (std::size_t i = 0; i < 9; ++i) { result[i] = {ids[rows[i]], scores[i]}; }
        std::sort(result.begin(), result.end(), [](const auto& x, const auto& y) {
            return x.score > y.score || (x.score == y.score && x.id < y.id);
        });
        return result;
    };
    const auto expected = sorted(a), actual = sorted(b);
    check(actual[0].id == 900 && actual[1].id == 2, "one-ULP leader and exact ties retain ID rule");
    for (std::size_t i = 0; i < 9; ++i) {
        check(actual[i].id == expected[i].id && std::bit_cast<std::uint32_t>(actual[i].score) ==
            std::bit_cast<std::uint32_t>(expected[i].score), "exact Top-K order/membership implication");
    }
}
} // namespace

int main() {
    if (!study::supported()) { std::cout << "SKIP: AVX2/FMA unavailable\n"; return 77; }
    if (!study::supported_environment()) { std::cout << "SKIP: require nearest, FTZ/DAZ off, masked exceptions\n"; return 77; }
    try {
        const auto saved = study::environment();
        _mm_setcsr(saved | 0x8040U);
        check(!study::supported_environment(), "reject FTZ/DAZ rather than change them silently");
        _mm_setcsr(saved);
        check(std::fesetround(FE_DOWNWARD) == 0 && !study::supported_environment(), "reject non-nearest rounding");
        check(std::fesetround(FE_TONEAREST) == 0, "restore rounding");
        dimensions_and_groups(); numerical_cases(); bounds(); ranking();
        std::cout << "PASS " << comparisons << " exact oracle/production/scalar/SIMD comparisons; MXCSR="
                  << study::environment() << "; rounding=" << std::fegetround() << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
