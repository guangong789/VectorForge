#include "scoring/candidate_scoring_simd.hpp"
#include "vectorforge/batched_executor.hpp"
#include "vectorforge/column_batch.hpp"
#include "vectorforge/embedding_buffer.hpp"
#include "vectorforge/validated_dataset.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <string_view>
#include <sys/auxv.h>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <xmmintrin.h>
#define VF_PROVIDER_X86 1
#else
#define VF_PROVIDER_X86 0
#endif

// Test-only copy of the existing executor TU with only its entry/scorer symbols
// renamed by CMake. Same validation, heap/allocation and error order, with the
// original generic scorer fixed as the oracle; no public API is added.
namespace vectorforge {
std::vector<ScoredRecord> execute_batched_scalar_control(const ColumnBatch&, const EmbeddingBuffer&,
    std::span<const RecordID>, std::span<const float>, std::span<const Predicate>, std::size_t, std::size_t);
std::vector<ScoredRecord> execute_batched_scalar_control(const ValidatedDataset&,
    std::span<const float>, std::span<const Predicate>, std::size_t, std::size_t);
}

namespace {
using namespace vectorforge;
using Count = std::size_t (*)();
Count provider_calls = nullptr;
std::size_t comparisons = 0;
void check(bool value, const char* message) { if (!value) { throw std::runtime_error(message); } }
struct Environment {
    std::fenv_t saved{}; int error = errno;
    Environment() { check(std::fegetenv(&saved) == 0, "save caller environment"); }
    ~Environment() { std::fesetenv(&saved); errno = error; }
};
struct State {
    int error, flags, rounding;
    unsigned csr = 0;
    unsigned short control = 0, status = 0;
};
State state() {
    const auto error = errno;
    State s{error, std::fetestexcept(FE_ALL_EXCEPT), std::fegetround()};
#if VF_PROVIDER_X86
    s.csr = _mm_getcsr(); asm volatile("fnstcw %0" : "=m"(s.control)); asm volatile("fnstsw %0" : "=am"(s.status));
#endif
    return s;
}
bool same(const State& a, const State& b) {
    return a.error == b.error && a.flags == b.flags && a.rounding == b.rounding && a.csr == b.csr &&
        a.control == b.control && (a.status & 0x3fU) == (b.status & 0x3fU);
}
struct Observed {
    std::array<float, 34> scores{};
    std::vector<ScoredRecord> records;
    std::size_t prefix = std::numeric_limits<std::size_t>::max(), calls = 0;
    const char* exception = "none";
    std::array<char, 192> diagnostic{};
    State final{};
};
template<class F> Observed observe(const std::fenv_t& start, F function) {
    const auto before = provider_calls ? provider_calls() : 0;
    check(std::fesetenv(&start) == 0, "establish identical initial state"); errno = E2BIG;
    Observed result; result.scores.fill(-99.25F);
    auto caught = [&](const char* type, const char* message) {
        result.final = state(); result.exception = type;
        std::strncpy(result.diagnostic.data(), message, result.diagnostic.size() - 1);
    };
    try { function(result); result.final = state(); }
    catch (const std::invalid_argument& e) { caught("invalid_argument", e.what()); }
    catch (const std::out_of_range& e) { caught("out_of_range", e.what()); }
    catch (const std::overflow_error& e) { caught("overflow_error", e.what()); }
    result.calls = provider_calls ? provider_calls() - before : 0;
    return result;
}
void equal(const Observed& a, const Observed& b) {
    check(a.prefix == b.prefix && std::strcmp(a.exception, b.exception) == 0 && a.diagnostic == b.diagnostic,
          "completed prefix/exception/diagnostic changed");
    check(same(a.final, b.final), "production FP state/control or errno differs from original scalar");
    for (std::size_t i = 0; i < a.scores.size(); ++i) {
        check(std::bit_cast<std::uint32_t>(a.scores[i]) == std::bit_cast<std::uint32_t>(b.scores[i]), "score bits/suffix/canary changed");
    }
    check(a.records.size() == b.records.size(), "query cardinality changed");
    for (std::size_t i = 0; i < a.records.size(); ++i) {
        check(a.records[i].id == b.records[i].id && std::bit_cast<std::uint32_t>(a.records[i].score) ==
            std::bit_cast<std::uint32_t>(b.records[i].score), "query IDs/order/score bits changed");
    }
    ++comparisons;
}

// False explicitly reports missing interception coverage, e.g. compiler-expanded
// std::fma under caller-supplied hardware target options; it is never a fake pass.
bool check_paths(bool replacement) {
    const float tiny = std::numeric_limits<float>::denorm_min();
    const ColumnBatch columns({std::vector<std::int32_t>(8, 1)});
    const EmbeddingBuffer buffer(8, 1, std::vector<float>(8, tiny));
    const std::array<RecordID, 8> ids{81, 18, 63, 36, 45, 54, 27, 72};
    const ValidatedDataset dataset(columns, buffer, ids);
    const std::array query{0.5F};
    const std::array none{Predicate{0, Comparison::equal, std::int32_t{2}}};
    const std::array invalid{Predicate{0, Comparison::equal, std::int64_t{1}}};
    std::array<std::size_t, 8> rows; std::iota(rows.begin(), rows.end(), 0);
    std::vector<float> overflowing(8, tiny); overflowing[1] = std::numeric_limits<float>::max();
    const EmbeddingBuffer overflow(8, 1, std::move(overflowing));
    const ValidatedDataset overflow_dataset(columns, overflow, ids);
    const std::array twice{2.0F};
    for (const auto incoming : {0, FE_UNDERFLOW | FE_INEXACT}) {
        std::feclearexcept(FE_ALL_EXCEPT);
#if VF_PROVIDER_X86
        asm volatile("fnclex"); _mm_setcsr(_mm_getcsr() & ~0x3fU);
#endif
        check(std::feraiseexcept(incoming) == 0, "seed incoming flags");
        std::fenv_t start; check(std::fegetenv(&start) == 0, "snapshot input environment");
        auto kernel = [&](const EmbeddingBuffer& b, std::span<const float> q, std::span<const std::size_t> indices) {
            const auto a = observe(start, [&](Observed& out) {
                out.prefix = detail::score_candidates_scalar(b, q, indices, std::span(out.scores).subspan(1, indices.size()));
            });
            const auto production = observe(start, [&](Observed& out) {
                out.prefix = detail::score_candidates(b, q, indices, std::span(out.scores).subspan(1, indices.size()));
            });
            equal(a, production); return std::pair{a, production};
        };
        const auto [expected, actual] = kernel(buffer, query, rows);
        check(expected.prefix == 8, "underflow fixture must complete all candidates");
        if (replacement && expected.calls == 0) { return false; }
        if (replacement) {
            check(expected.final.error == ERANGE && actual.calls != 0,
                  "replacement must be invoked by both generic and production scoring");
        }
        const auto partial = kernel(overflow, twice, rows);
        check(partial.first.prefix == 1, "overflow preserves partial completed prefix");
        (void)kernel(buffer, {}, rows);
        auto bad = rows; bad.back() = std::numeric_limits<std::size_t>::max();
        (void)kernel(buffer, query, bad);
        (void)kernel(overflow, twice, bad); // Earlier overflow precedes later bad row.
        auto queries = [&](const EmbeddingBuffer& b, const ValidatedDataset& admitted, std::span<const float> q,
                           std::span<const Predicate> predicates, std::size_t k, std::size_t batch, bool scores_expected) {
            for (const bool trusted : {false, true}) {
                const auto control = observe(start, [&](Observed& out) {
                    out.records = trusted ? execute_batched_scalar_control(admitted, q, predicates, k, batch)
                        : execute_batched_scalar_control(columns, b, ids, q, predicates, k, batch);
                });
                const auto production = observe(start, [&](Observed& out) {
                    out.records = trusted ? execute_batched(admitted, q, predicates, k, batch)
                        : execute_batched(columns, b, ids, q, predicates, k, batch);
                });
                equal(control, production);
                if (replacement) {
                    check(scores_expected ? control.calls != 0 && production.calls != 0
                                          : control.calls == 0 && production.calls == 0,
                          "query provider invocation/K=0/no-match validation changed");
                    if (scores_expected && &b == &buffer) {
                        check(control.final.error == ERANGE && production.final.error == ERANGE, "public queries must retain provider errno");
                    }
                }
            }
        };
        queries(buffer, dataset, query, {}, 3, 8, true);
        queries(buffer, dataset, query, {}, 3, 3, true);
        queries(buffer, dataset, query, {}, 0, 8, false);
        queries(buffer, dataset, query, none, 3, 8, false);
        queries(buffer, dataset, query, invalid, 3, 8, false);
        queries(buffer, dataset, {}, {}, 3, 8, false);
        queries(buffer, dataset, query, {}, 3, 0, false);
        queries(overflow, overflow_dataset, twice, {}, 1, 8, true);
    }
    return true;
}
} // namespace

int main(int argc, char** argv) {
    try {
        check(argc == 2, "use --normal or --replacement");
        const bool replacement = std::string_view(argv[1]) == "--replacement";
        check(replacement || std::string_view(argv[1]) == "--normal", "unknown provider mode");
        if (getauxval(AT_BASE) == 0) { std::cout << "SKIP: no dynamic ELF loader; no interposition coverage\n"; return 77; }
        provider_calls = reinterpret_cast<Count>(dlsym(RTLD_DEFAULT, "vectorforge_test_fmaf_calls"));
        check((provider_calls != nullptr) == replacement, "replacement fixture not loaded as requested");
        if (replacement) {
            const auto ready = reinterpret_cast<bool (*)()>(dlsym(RTLD_DEFAULT, "vectorforge_test_fmaf_ready"));
            check(ready != nullptr && ready(), "fixture cannot resolve real fmaf");
        }
        Environment restore;
        std::fenv_t masked; check(std::feholdexcept(&masked) == 0 && std::fesetround(FE_TONEAREST) == 0, "establish masked nearest test environment");
#if VF_PROVIDER_X86
        _mm_setcsr((_mm_getcsr() & ~0xe07fU) | 0x1f80U);
#endif
        if (!check_paths(replacement)) { std::cout << "SKIP: generic scorer does not call replaceable fmaf (compiler/direct binding); no substitution coverage\n"; return 77; }
        std::cout << "PASS " << comparisons << " scalar/provider/public-query comparisons; replacement=" << replacement << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
