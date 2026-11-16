#include "scoring/candidate_scoring_simd.hpp"
#include "vectorforge/batched_executor.hpp"
#include "vectorforge/column_batch.hpp"
#include "vectorforge/embedding_buffer.hpp"
#include "vectorforge/scalar_reference.hpp"
#include "vectorforge/validated_dataset.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <new>
#include <numeric>
#include <random>
#include <string>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <xmmintrin.h>
#define VF_TEST_X86 1
#else
#define VF_TEST_X86 0
#endif

namespace failure { bool armed = false; std::size_t hits = 0; }
void* operator new(std::size_t bytes) {
    if (failure::armed && bytes == sizeof(vectorforge::ScoredRecord)) {
        failure::armed = false; ++failure::hits; throw std::bad_alloc();
    }
    if (auto* p = std::malloc(bytes == 0 ? 1 : bytes)) { return p; }
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {
using namespace vectorforge;
using Kernel = std::size_t (*)(const EmbeddingBuffer&, std::span<const float>, std::span<const std::size_t>, std::span<float>);
constexpr std::array<Kernel, 3> paths{detail::score_candidates_scalar, detail::score_candidates_auto, detail::score_candidates};
std::size_t comparisons = 0;
void check(bool condition, const char* message) { if (!condition) { throw std::runtime_error(message); } }
template<class E, class F> void throws(F f) {
    try { f(); } catch (const E&) { return; }
    throw std::runtime_error("expected exception missing");
}
struct Environment {
    std::fenv_t state{};
    int error = errno;
    Environment() { check(std::fegetenv(&state) == 0, "save FP environment"); }
    ~Environment() { std::fesetenv(&state); errno = error; }
};

void compare(const EmbeddingBuffer& buffer, std::span<const float> query, std::span<const std::size_t> rows) {
    std::array<float, 34> expected;
    expected.fill(-99.25F);
    const auto prefix = detail::score_candidates_scalar(buffer, query, rows, std::span(expected).subspan(1, rows.size()));
    for (const auto path : paths) {
        std::array<float, 34> actual;
        actual.fill(-99.25F);
        check(path(buffer, query, rows, std::span(actual).subspan(1, rows.size())) == prefix, "ordered finite prefix");
        for (std::size_t i = 0; i < actual.size(); ++i) {
            check(std::bit_cast<std::uint32_t>(actual[i]) == std::bit_cast<std::uint32_t>(expected[i]),
                  "exact bits, row association, unwritten suffix and canaries");
        }
        ++comparisons;
    }
}

void groups_and_numerics() {
    std::mt19937 random(0x56463108U);
    auto value = [&] { return static_cast<float>(static_cast<int>(random() % 2001) - 1000) / 997.0F; };
    for (const auto d : {1U, 2U, 3U, 7U, 9U, 32U, 128U, 129U, 768U}) {
        std::vector<float> values(97 * d), query(d);
        std::generate(values.begin(), values.end(), value); std::generate(query.begin(), query.end(), value);
        const EmbeddingBuffer buffer(97, d, std::move(values));
        for (const auto n : {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 15U, 16U, 17U, 31U, 32U}) {
            std::array<std::size_t, 32> rows{};
            for (int mode = 0; mode < 3; ++mode) {
                for (std::size_t i = 0; i < n; ++i) { rows[i] = mode == 0 ? 97 - n + i : mode == 1 ? (i * 37 + 13) % 97 : 96; }
                compare(buffer, query, std::span(rows).first(n));
            }
        }
    }
    const float a = std::ldexp(1.0F, -23), tiny = std::numeric_limits<float>::denorm_min();
    std::vector<float> data(33 * 3);
    for (std::size_t r = 0; r < 33; ++r) { data[r * 3] = 1; data[r * 3 + 1] = 1 - a; data[r * 3 + 2] = r % 2 ? tiny : -0.0F; }
    const EmbeddingBuffer buffer(33, 3, std::move(data));
    const std::array query{-1.0F, 1 + a, 1.0F};
    std::array<std::size_t, 32> rows; std::iota(rows.begin(), rows.end(), 1);
    compare(buffer, query, rows);
    const EmbeddingBuffer zeros(9, 1, {0, -0.0F, tiny, -tiny, 0, -0.0F, tiny, -tiny, 0});
    for (const auto q : {0.0F, -0.0F, 0.5F, 1.0F, -1.0F}) {
        const std::array one{q}; std::iota(rows.begin(), rows.end(), 0);
        compare(zeros, one, std::span(rows).first(9));
    }
}

void errors_and_addresses() {
    const float largest = std::numeric_limits<float>::max();
    std::array<std::size_t, 32> rows; std::iota(rows.begin(), rows.end(), 0);
    const std::array twice{2.0F};
    for (std::size_t failed = 0; failed < 32; ++failed) {
        std::vector<float> data(32, 1); data[failed] = largest;
        const EmbeddingBuffer buffer(32, 1, std::move(data)); compare(buffer, twice, rows);
    }
    std::vector<float> data(9 * 3, 0); data[1 * 3 + 2] = largest; data[6 * 3] = largest;
    const EmbeddingBuffer different_dimensions(9, 3, std::move(data));
    const std::array query{2.0F, 1.0F, 2.0F};
    compare(different_dimensions, query, std::span(rows).first(9));
    const EmbeddingBuffer buffer(9, 1, std::vector<float>(9, 1));
    std::array<float, 33> out{}; std::array<std::size_t, 33> oversized{};
    for (const auto path : paths) {
        throws<std::invalid_argument>([&] { (void)path(buffer, {}, rows, std::span(out).first(32)); });
        throws<std::invalid_argument>([&] { (void)path(buffer, twice, oversized, out); });
        throws<std::invalid_argument>([&] { (void)path(buffer, twice, std::span(rows).first(8), {}); });
        auto bad = rows; bad[8] = std::numeric_limits<std::size_t>::max();
        throws<std::out_of_range>([&] { (void)path(buffer, twice, std::span(bad).first(9), std::span(out).first(9)); });
        std::vector<float> overflow_data(9, 1); overflow_data[0] = largest;
        const EmbeddingBuffer overflow(9, 1, std::move(overflow_data));
        check(path(overflow, twice, std::span(bad).first(9), std::span(out).first(9)) == 0,
              "earlier intermediate overflow must precede invalid later address");
    }
    constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float);
    check(detail::simd_row_addressable(limit, 1) && !detail::simd_row_addressable(limit + 1, 1), "byte-distance boundary");
    check(!detail::simd_row_addressable(limit, 2) && !detail::simd_row_addressable(0, 0), "row multiplication bounds");
    check(!detail::simd_row_addressable(0, std::numeric_limits<std::size_t>::max()), "dimension overflow bound");
    if constexpr (sizeof(std::size_t) == 8) { check(detail::simd_row_addressable(std::size_t{1} << 31, 1), "not restricted to signed i32 offsets"); }
}

void near_tie_queries() {
    std::vector<float> values(33 * 8, 0);
    std::vector<std::int32_t> flags(33);
    std::array<RecordID, 33> ids;
    for (std::size_t row = 0; row < 33; ++row) {
        values[row * 8] = 1; flags[row] = row % 3 != 0 ? 1 : 0; ids[row] = 1000 - row * 17;
    }
    values[8 * 8] = std::nextafter(1.0F, 2.0F);
    values[1] = std::ldexp(1.0F, -24); values[5] = std::ldexp(1.0F, -24);
    const ColumnBatch columns({std::move(flags)});
    const EmbeddingBuffer embeddings(33, 8, std::move(values));
    const ValidatedDataset dataset(columns, embeddings, ids);
    const std::array query{1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
    const std::array sparse{Predicate{0, Comparison::equal, std::int32_t{1}}};
    const std::array none{Predicate{0, Comparison::equal, std::int32_t{2}}};
    for (const auto predicates : {std::span<const Predicate>{}, std::span<const Predicate>(sparse), std::span<const Predicate>(none)}) {
        for (const auto k : {0U, 1U, 3U, 99U}) {
            const auto expected = execute_scalar_reference(dataset, query, predicates, k);
            if (k != 0 && predicates.data() != none.data()) { check(expected[0].id == ids[8], "one-ULP leader beats lower IDs"); }
            for (const auto batch : {7U, 8U, 9U, 31U, 32U, 33U, 256U}) {
                const auto a = execute_batched(dataset, query, predicates, k, batch);
                const auto b = execute_batched(columns, embeddings, ids, query, predicates, k, batch);
                for (const auto& actual : {a, b}) {
                    check(actual.size() == expected.size(), "query cardinality at K=0/empty/full/tail windows");
                    for (std::size_t i = 0; i < actual.size(); ++i) {
                        check(actual[i].id == expected[i].id && std::bit_cast<std::uint32_t>(actual[i].score) ==
                            std::bit_cast<std::uint32_t>(expected[i].score), "near-tie order, mapping and exact scores");
                    }
                    ++comparisons;
                }
            }
        }
    }
}

void changing_environment() {
    Environment restore;
    const EmbeddingBuffer buffer(9, 1, std::vector<float>(9, 1));
    const std::array query{1.0F}; std::array<std::size_t, 9> rows{}; std::array<float, 9> out{};
    const ColumnBatch columns({std::vector<std::int32_t>(9, 1)});
    const std::array<RecordID, 9> ids{90, 70, 80, 30, 20, 40, 60, 50, 10};
    const ValidatedDataset dataset(columns, buffer, ids);
    auto query_exact = [&] {
        const auto a = execute_batched(dataset, query, {}, 1, 9);
        const auto b = execute_batched(columns, buffer, ids, query, {}, 1, 9);
        for (const auto& result : {a, b}) {
            check(result.size() == 1 && result[0].id == 10 && std::bit_cast<std::uint32_t>(result[0].score) ==
                std::bit_cast<std::uint32_t>(1.0F), "repeated public queries honor changed FP environment");
        }
    };
    for (const auto mode : {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        check(std::fesetround(mode) == 0, "change rounding mode");
        check(!detail::simd_scoring_eligible(buffer, query, rows, out), "FP mode cannot be cached");
        compare(buffer, query, rows);
        query_exact();
    }
    check(std::fesetround(FE_TONEAREST) == 0, "restore nearest rounding");
#if VF_TEST_X86
    const auto saved = _mm_getcsr();
    for (const auto mask : {0x8000U, 0x40U, 0x8040U}) {
        _mm_setcsr(saved | mask);
        check(!detail::simd_scoring_eligible(buffer, query, rows, out), "FTZ/DAZ force generic fallback");
        compare(buffer, query, rows);
        query_exact();
    }
    _mm_setcsr(saved);
    for (unsigned bit = 7; bit <= 12; ++bit) {
        _mm_setcsr((saved & ~0x3fU) & ~(1U << bit)); // clear pending flags before unmasking
        check(!detail::simd_scoring_eligible(buffer, query, rows, out), "unmasked SSE exception forces scalar");
        compare(buffer, query, rows); // exact normal arithmetic, no trap
        query_exact();
    }
    _mm_setcsr(saved);
    unsigned short control;
    asm volatile("fnstcw %0" : "=m"(control));
    const auto unmasked = static_cast<unsigned short>(control & ~1U);
    asm volatile("fldcw %0" : : "m"(unmasked));
    check(!detail::simd_scoring_eligible(buffer, query, rows, out), "x87 masks must also be checked");
    asm volatile("fldcw %0" : : "m"(control));
#endif
    check(detail::simd_scoring_eligible(buffer, query, rows, out) ==
        (detail::simd_hardware_supported() && detail::simd_platform_supported()), "restored environment enables only supported platform/hardware");
    query_exact();
}

void observable_side_effect_gate() {
    Environment restore;
    const float largest = std::numeric_limits<float>::max(), tiny = std::numeric_limits<float>::denorm_min();
    std::vector<float> data(8 * 3, 0); data[0] = 1; data[2] = largest; data[3] = tiny;
    const EmbeddingBuffer buffer(8, 3, std::move(data));
    const std::array query{0.5F, 1.0F, 2.0F};
    std::array<std::size_t, 8> rows; std::iota(rows.begin(), rows.end(), 0); std::array<float, 8> out;
    struct Observed { std::size_t prefix; int flags; int error; unsigned csr; unsigned short x87; };
    auto observe = [&](Kernel path) {
        std::feclearexcept(FE_ALL_EXCEPT); errno = E2BIG; out.fill(-99);
#if VF_TEST_X86
        asm volatile("fnclex"); _mm_setcsr(_mm_getcsr() & ~0x3fU);
#endif
        const auto prefix = path(buffer, query, rows, out);
        const auto error = errno; const auto flags = std::fetestexcept(FE_ALL_EXCEPT);
        unsigned csr = 0; unsigned short x87 = 0;
#if VF_TEST_X86
        csr = _mm_getcsr(); asm volatile("fnstsw %0" : "=am"(x87));
#endif
        check(std::all_of(out.begin(), out.end(), [](float x) { return x == -99; }), "no outputs after first overflow");
        return Observed{prefix, flags, error, csr, x87};
    };
    const auto a = observe(detail::score_candidates_scalar), b = observe(detail::score_candidates_auto);
    check(a.prefix == 0 && b.prefix == 0, "C++ error prefix still agrees");
    check(a.flags == b.flags && a.error == b.error && a.csr == b.csr && (a.x87 & 0x3fU) == (b.x87 & 0x3fU),
          "isolated dispatch must preserve standard and raw per-unit flags/errno");
    std::cout << "FP STATE MATCH: scalar flags=" << a.flags << " auto flags=" << b.flags << " errno=" << a.error << '/' << b.error << '\n';
    const auto production = observe(detail::score_candidates);
    check(production.flags == a.flags && production.error == a.error, "production default preserves original side effects");
    // Preserve already-set flags/errno in an ordinary finite invocation too.
    const EmbeddingBuffer finite(8, 1, std::vector<float>(8, 1)); const std::array one{1.0F};
    for (const auto path : paths) {
        std::feclearexcept(FE_ALL_EXCEPT); std::feraiseexcept(FE_DIVBYZERO); errno = E2BIG;
        check(path(finite, one, rows, out) == 8 && (std::fetestexcept(FE_DIVBYZERO) & FE_DIVBYZERO) && errno == E2BIG,
              "ordinary scores do not erase caller status or errno");
    }
}

void allocation_precedence() {
    for (const bool first : {false, true}) {
        const ColumnBatch columns({std::vector<std::int32_t>(9, 1)});
        std::vector<float> data(9, 1); data[first ? 0 : 1] = std::numeric_limits<float>::max();
        const EmbeddingBuffer embeddings(9, 1, std::move(data));
        const std::array<RecordID, 9> ids{8, 7, 6, 5, 4, 3, 2, 1, 0}; const std::array query{2.0F};
        const ValidatedDataset dataset(columns, embeddings, ids);
        failure::hits = 0; failure::armed = true; bool correct = false;
        try { (void)execute_batched(dataset, query, {}, 1, 9); }
        catch (const std::bad_alloc&) { correct = !first; }
        catch (const std::overflow_error& e) { correct = first && std::string(e.what()) == "batched dot product produced a non-finite score"; }
        catch (...) { failure::armed = false; throw; }
        failure::armed = false;
        check(correct && failure::hits == (first ? 0U : 1U), "full SIMD group preserves allocation/error precedence");
    }
}
} // namespace
int main() {
    try {
        groups_and_numerics(); errors_and_addresses(); near_tie_queries(); changing_environment(); observable_side_effect_gate(); allocation_precedence();
        std::cout << "PASS " << comparisons << " exact dispatch comparisons; native=" << detail::simd_hardware_supported()
                  << "; legacy_marker=" << VECTORFORGE_TEST_SIMD_DISPATCH << '\n';
        return 0;
    } catch (const std::exception& e) { failure::armed = false; std::cerr << e.what() << '\n'; return 1; }
}
