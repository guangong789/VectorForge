#include "fp_state.hpp"
#include "scoring/candidate_scoring_simd.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <sys/utsname.h>
#include <vector>
#include <xmmintrin.h>
#if defined(__GLIBC__)
#include <gnu/libc-version.h>
#endif

namespace {
using namespace vectorforge;
namespace study = benchmark::fp_state;
constexpr std::array<study::Kernel, 3> kernels{detail::score_candidates_scalar, study::unisolated_scores, study::isolated_scores};
constexpr std::array<const char*, 3> names{"scalar", "existing_simd", "isolated"};
std::size_t comparisons = 0, raw_mismatches = 0;
void check(bool value, const char* message) { if (!value) { throw std::runtime_error(message); } }

struct SavedEnvironment {
    std::fenv_t state{};
    int error = errno;
    SavedEnvironment() { check(std::fegetenv(&state) == 0, "save outer FP environment"); }
    ~SavedEnvironment() { std::fesetenv(&state); errno = error; }
};
struct State {
    int flags, rounding, error;
    unsigned csr;
    unsigned short control, status;
};
State state() {
    const int error = errno;
    State s{std::fetestexcept(FE_ALL_EXCEPT), std::fegetround(), error, _mm_getcsr(), 0, 0};
    asm volatile("fnstcw %0" : "=m"(s.control));
    asm volatile("fnstsw %0" : "=am"(s.status));
    return s;
}
bool same_controls(const State& a, const State& b) {
    return a.rounding == b.rounding && a.control == b.control && (a.csr & ~0x3fU) == (b.csr & ~0x3fU);
}
bool same_state(const State& a, const State& b) {
    return same_controls(a, b) && a.flags == b.flags && a.error == b.error &&
        a.csr == b.csr && (a.status & 0x3fU) == (b.status & 0x3fU);
}
void print_state(const State& s) {
    std::cout << "fe=" << s.flags << ";sse=" << (s.csr & 0x3fU) << ";x87=" << (s.status & 0x3fU)
              << ";round=" << s.rounding << ";ftz=" << bool(s.csr & 0x8000U) << ";daz=" << bool(s.csr & 0x40U)
              << ";sse_masks=" << ((s.csr >> 7) & 0x3fU) << ";x87_masks=" << (s.control & 0x3fU) << ";errno=" << s.error;
}
struct Initial {
    std::fenv_t environment{};
    State view{};
};
Initial initial(const std::fenv_t& outer, int flags) {
    check(std::fesetenv(&outer) == 0, "restore starting controls");
    // Test setup only: clear all six raw flags, including MXCSR denormal operand
    // (not represented by glibc FE_ALL_EXCEPT). Never used at scorer exit.
    asm volatile("fnclex");
    _mm_setcsr(_mm_getcsr() & ~0x3fU);
    check(std::feraiseexcept(flags) == 0, "seed caller exception flags");
    errno = E2BIG;
    Initial result;
    result.view = state();
    check(std::fegetenv(&result.environment) == 0, "capture identical initial environment");
    check(same_state(result.view, state()), "fegetenv must preserve observed controls and flags");
    return result;
}
struct Case {
    std::string name;
    EmbeddingBuffer buffer;
    std::vector<float> query;
    std::vector<std::size_t> rows;
    bool checked = false;
};
Case fixture(std::string name, std::size_t n, std::size_t d, std::vector<float> values, std::vector<float> query,
             bool checked = false) {
    std::vector<std::size_t> rows(n); std::iota(rows.begin(), rows.end(), 0);
    return {std::move(name), EmbeddingBuffer(n, d, std::move(values)), std::move(query), std::move(rows), checked};
}
Case counterexample(std::string name, std::size_t d = 3, bool remove_later = false) {
    std::vector<float> data(8 * d, 0), query(d, 1);
    query[0] = 0.5F; query[d - 1] = 2;
    data[0] = 1; data[d - 1] = std::numeric_limits<float>::max();
    if (!remove_later) { data[d] = std::numeric_limits<float>::denorm_min(); }
    return fixture(std::move(name), 8, d, std::move(data), std::move(query));
}
struct Observed {
    std::array<float, 34> scores{};
    std::size_t prefix = std::numeric_limits<std::size_t>::max();
    const char* exception = "none";
    std::array<char, 192> diagnostic{};
    State final{};
};
Observed observe(study::Kernel kernel, const Case& c, const Initial& start) {
    check(std::fesetenv(&start.environment) == 0, "establish same initial state"); errno = start.view.error;
    Observed result; result.scores.fill(-99.25F);
    auto caught = [&](const char* type, const char* message) {
        result.final = state(); result.exception = type;
        std::strncpy(result.diagnostic.data(), message, result.diagnostic.size() - 1);
    };
    try {
        result.prefix = kernel(c.buffer, c.query, c.rows, std::span(result.scores).subspan(1, c.rows.size()));
        if (c.checked && result.prefix != c.rows.size()) {
            throw std::overflow_error("batched dot product produced a non-finite score");
        }
        result.final = state();
    } catch (const std::invalid_argument& e) { caught("invalid_argument", e.what()); }
      catch (const std::out_of_range& e) { caught("out_of_range", e.what()); }
      catch (const std::overflow_error& e) { caught("overflow_error", e.what()); }
    return result;
}
void same_result(const Observed& a, const Observed& b) {
    check(a.prefix == b.prefix && std::strcmp(a.exception, b.exception) == 0 && a.diagnostic == b.diagnostic,
          "completed prefix/C++ exception/diagnostic mismatch");
    for (std::size_t i = 0; i < a.scores.size(); ++i) {
        check(std::bit_cast<std::uint32_t>(a.scores[i]) == std::bit_cast<std::uint32_t>(b.scores[i]),
              "exact score bits, unwritten suffix or canary mismatch");
    }
}
std::vector<Case> cases() {
    const float tiny = std::numeric_limits<float>::denorm_min(), largest = std::numeric_limits<float>::max();
    std::vector<Case> result;
    result.push_back(counterexample("original"));
    result.push_back(counterexample("later_lane_removed", 3, true));
    result.push_back(fixture("legitimate_underflow_success", 8, 1, std::vector<float>(8, tiny), {0.5F}));
    std::vector<float> data(16);
    for (std::size_t i = 0; i < 8; ++i) { data[i * 2] = 1; data[i * 2 + 1] = std::ldexp(1.0F, -24); }
    result.push_back(fixture("inexact_success", 8, 2, std::move(data), {1, 1}));
    const float a = std::ldexp(1.0F, -23); data.assign(16, 0);
    for (std::size_t i = 0; i < 8; ++i) { data[i * 2] = 1; data[i * 2 + 1] = 1 - a; }
    result.push_back(fixture("fused_cancellation", 8, 2, std::move(data), {-1, 1 + a}));
    result.push_back(fixture("signed_zero_subnormal_tail", 9, 1, {0, -0.0F, tiny, -tiny, 0, -0.0F, tiny, -tiny, 0}, {-1}));
    data.assign(16 * 3, 0); data[1] = 0.1F; data[8 * 3 + 2] = largest; data[9 * 3] = tiny;
    result.push_back(fixture("successful_group_before_failure", 16, 3, std::move(data), {0.5F, 0.1F, 2}));
    data.assign(24 * 3, 0); data[0] = tiny; data[8 * 3 + 1] = 0.1F; data[16 * 3 + 2] = largest; data[17 * 3] = tiny;
    result.push_back(fixture("two_prior_groups_keep_flags", 24, 3, std::move(data), {0.5F, 0.1F, 2}));
    data.assign(9 * 3, 0); data[1] = tiny; data[1 * 3 + 2] = largest; data[6 * 3] = largest;
    result.push_back(fixture("multiple_failures_ordered_cpp_error", 9, 3, std::move(data), {2, 0.5F, 2}, true));
    // Diagnostic outside admitted finite-input domain, to exercise discarding
    // speculative INVALID too; it is not evidence about public query admission.
    data.assign(16 * 3, 0); data[1] = tiny; data[8 * 3 + 2] = largest; data[9 * 3] = std::numeric_limits<float>::infinity();
    result.push_back(fixture("outside_contract_speculative_invalid", 16, 3, std::move(data), {0, 0.5F, 2}));
    auto partial = fixture("scalar_partial_group", 8, 1, std::vector<float>(8, tiny), {0.5F});
    partial.rows.resize(7); result.push_back(std::move(partial));
    result.push_back(fixture("empty", 0, 1, {}, {1}));
    auto invalid = fixture("invalid_row_after_finite_prefix", 8, 1, std::vector<float>(8, 1), {1});
    invalid.rows.back() = 100; result.push_back(std::move(invalid));
    auto shape = fixture("invalid_query_shape", 8, 2, std::vector<float>(16, 1), {1});
    result.push_back(std::move(shape));
    data.resize(33 * 9);
    for (std::size_t i = 0; i < data.size(); ++i) { data[i] = static_cast<float>(static_cast<int>(i % 127) - 63) / 97.0F; }
    auto sparse = fixture("four_groups_sparse_repeated", 33, 9, std::move(data), std::vector<float>(9, 0.3F));
    sparse.rows.resize(32);
    for (std::size_t i = 0; i < 32; ++i) { sparse.rows[i] = i % 3 == 0 ? 32 : (i * 17 + 2) % 33; }
    result.push_back(std::move(sparse));
    return result;
}

void localization(const std::fenv_t& outer) {
    const auto start = initial(outer, 0);
    const auto c = counterexample("localize");
    for (const auto row : {0U, 1U}) {
        check(std::fesetenv(&start.environment) == 0, "localization initial state"); errno = start.view.error;
        float previous = 0;
        for (std::size_t d = 0; d < c.query.size(); ++d) {
            previous = study::scalar_step(c.query[d], c.buffer.row(row)[d], previous);
            const auto after = state();
            std::cout << "trace row=" << row << " d=" << d << " bits=" << std::bit_cast<std::uint32_t>(previous) << ' ';
            print_state(after); std::cout << '\n';
            if (!std::isfinite(previous)) { break; }
        }
    }
}

void semantic_tests(const std::fenv_t& outer) {
    localization(outer);
    const auto fixtures = cases();
    constexpr std::array incoming{0, FE_UNDERFLOW, FE_INEXACT, FE_UNDERFLOW | FE_INEXACT, FE_ALL_EXCEPT};
    for (const auto flags : incoming) {
        const auto start = initial(outer, flags);
        for (const auto& c : fixtures) {
            const auto expected = observe(kernels[0], c, start);
            for (std::size_t k = 0; k < kernels.size(); ++k) {
                for (int repeat = 0; repeat < 3; ++repeat) {
                    const auto actual = observe(kernels[k], c, start);
                    same_result(expected, actual);
                    check(same_controls(start.view, actual.final), "rounding/FTZ/DAZ/masks/precision controls changed");
                    if (k != 1) { check(same_state(expected.final, actual.final), "full final FP flags/unit placement/errno mismatch"); }
                    else if (!same_state(expected.final, actual.final)) { ++raw_mismatches; }
                    ++comparisons;
                    if (repeat == 0) {
                        std::cout << "case=" << c.name << " seed=" << flags << " path=" << names[k] << " initial{";
                        print_state(start.view); std::cout << "} final{"; print_state(actual.final);
                        std::cout << "} prefix=" << actual.prefix << " exception=" << actual.exception
                                  << " diagnostic=" << actual.diagnostic.data() << " bits_equal=1\n";
                    }
                }
            }
            for (int repeat = 0; repeat < 3; ++repeat) {
                const auto integrated = observe(detail::score_candidates_auto, c, start);
                same_result(expected, integrated);
                check(same_state(expected.final, integrated.final) && same_controls(start.view, integrated.final),
                      "production-internal integrated state/errno/control mismatch");
                ++comparisons;
            }
        }
    }
    check(raw_mismatches != 0, "original speculative side-effect mismatch must reproduce");
    // Sequential calls inside a caller-owned outer checkpoint. Inner failure
    // must keep flags produced by the earlier successful call, with no globals.
    const auto sequence = [&](study::Kernel kernel, const Initial& start) {
        std::fesetenv(&start.environment); errno = start.view.error;
        std::fenv_t outer_checkpoint; check(std::fegetenv(&outer_checkpoint) == 0, "nested outer checkpoint");
        std::array<float, 8> output{};
        const auto& first = fixtures[2]; const auto& second = fixtures[0];
        check(kernel(first.buffer, first.query, first.rows, output) == 8, "first sequential call");
        check(kernel(second.buffer, second.query, second.rows, output) == 0, "inner failing call");
        const auto end = state();
        check(std::fesetenv(&outer_checkpoint) == 0, "restore caller-owned checkpoint");
        check(same_controls(start.view, state()), "nested caller control restoration");
        return end;
    };
    for (const auto flags : incoming) {
        const auto start = initial(outer, flags);
        check(same_state(sequence(kernels[0], start), sequence(kernels[2], start)), "sequential/nested-state composition");
        ++comparisons;
    }
    // Unsupported controls must pass through unchanged via the generic fallback.
    for (const auto mode : {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        auto start = initial(outer, 0); check(std::fesetround(mode) == 0, "test rounding mode");
        std::fegetenv(&start.environment); start.view = state();
        const auto a = observe(kernels[0], fixtures[3], start), b = observe(kernels[2], fixtures[3], start);
        same_result(a, b); check(same_state(a.final, b.final) && same_controls(start.view, b.final), "rounding fallback changed state");
        ++comparisons;
    }
    for (const auto bits : {0x40U, 0x8000U, 0x8040U}) {
        auto start = initial(outer, 0); _mm_setcsr(_mm_getcsr() | bits);
        std::fegetenv(&start.environment); start.view = state();
        const auto a = observe(kernels[0], fixtures[2], start), b = observe(kernels[2], fixtures[2], start);
        same_result(a, b); check(same_state(a.final, b.final) && same_controls(start.view, b.final), "FTZ/DAZ fallback changed state");
        ++comparisons;
    }
    std::fesetenv(&outer);
    std::cout << "PASS comparisons=" << comparisons << " reproduced_existing_state_mismatches=" << raw_mismatches << '\n';
}

void metadata() {
    std::string cpu; std::ifstream info("/proc/cpuinfo");
    while (std::getline(info, cpu)) { if (cpu.starts_with("model name")) { break; } }
    utsname os{}; uname(&os);
    Dl_info symbol{}; const auto* function = dlsym(RTLD_DEFAULT, "fmaf");
    const bool located = function != nullptr && dladdr(function, &symbol) != 0;
    const auto offset = located ? reinterpret_cast<std::uintptr_t>(function) - reinterpret_cast<std::uintptr_t>(symbol.dli_fbase) : 0;
    std::cout << "metadata compiler=" << __VERSION__ << " config=" << VECTORFORGE_FP_STUDY_CONFIG
              << " sanitized=" << VECTORFORGE_FP_STUDY_SANITIZED << " cpu=" << cpu << " os=" << os.release;
#if defined(__GLIBC__)
    std::cout << " glibc=" << gnu_get_libc_version();
#endif
    std::cout << " fmaf_file=" << (located ? symbol.dli_fname : "unknown") << " fmaf_offset=" << offset
              << " generic=-frounding-math,-fno-lto native=-march=x86-64,-mavx2,-mfma,-frounding-math,-fno-lto\n";
}

void performance(const std::fenv_t& outer) {
    check(!VECTORFORGE_FP_STUDY_SANITIZED && std::string_view(VECTORFORGE_FP_STUDY_CONFIG) == "Release", "timing requires unsanitized Release");
    std::vector<Case> workloads;
    for (const auto d : {32U, 128U, 768U}) {
        std::vector<float> values(8 * d), query(d);
        for (std::size_t i = 0; i < values.size(); ++i) { values[i] = static_cast<float>(static_cast<int>(i % 127) - 63) / 97.0F; }
        for (std::size_t i = 0; i < query.size(); ++i) { query[i] = static_cast<float>(i % 13) / 19.0F; }
        workloads.push_back(fixture("success_d" + std::to_string(d), 8, d, std::move(values), std::move(query)));
    }
    workloads.push_back(counterexample("failure_d32", 32));
    constexpr std::size_t repeats = 1024, warmup = 3, rounds = 9;
    std::cout << "PERF_HEADER case,path,d,repeats,median_ns_group,min_ns_group,max_ns_group,samples_ns_group,fe_flags,sse_flags,x87_flags,errno\n";
    for (const auto& c : workloads) {
        const auto start_state = initial(outer, 0);
        const auto expected = observe(kernels[0], c, start_state);
        std::array<std::vector<double>, 3> samples;
        auto sample = [&](std::size_t k, bool measured) {
            std::array<float, 34> output; output.fill(-99.25F);
            std::fesetenv(&start_state.environment); errno = start_state.view.error;
            std::size_t prefix = 0;
            const auto begin = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < repeats; ++i) {
                prefix = kernels[k](c.buffer, c.query, c.rows, std::span(output).subspan(1, 8));
            }
            const auto end = std::chrono::steady_clock::now();
            const auto final_state = state(); // before floating-point timing arithmetic
            Observed actual; actual.scores = output; actual.prefix = prefix; actual.final = final_state;
            same_result(expected, actual);
            if (k != 1) { check(same_state(expected.final, final_state), "timed semantic gate failed"); }
            if (measured) { samples[k].push_back(std::chrono::duration<double, std::nano>(end - begin).count() / repeats); }
        };
        for (std::size_t r = 0; r < warmup + rounds; ++r) {
            for (std::size_t i = 0; i < kernels.size(); ++i) { sample((r + i) % kernels.size(), r >= warmup); }
        }
        for (std::size_t k = 0; k < kernels.size(); ++k) {
            auto sorted = samples[k]; std::sort(sorted.begin(), sorted.end());
            const auto final = observe(kernels[k], c, start_state).final;
            std::cout << "PERF " << c.name << ',' << names[k] << ',' << c.query.size() << ',' << repeats << ','
                << sorted[rounds / 2] << ',' << sorted.front() << ',' << sorted.back() << ',';
            for (const auto ns : samples[k]) { std::cout << ns << ';'; }
            std::cout << ',' << final.flags << ',' << (final.csr & 0x3fU) << ',' << (final.status & 0x3fU) << ',' << final.error << '\n';
        }
    }
}
} // namespace
int main(int argc, char** argv) {
    if (!detail::simd_platform_supported() || !detail::simd_hardware_supported() || !detail::simd_environment_supported()) {
        std::cout << "SKIP: require supported platform, AVX2/FMA, nearest, FTZ/DAZ off, masked exceptions\n"; return 77;
    }
    try {
        const bool timed = argc == 2 && std::string_view(argv[1]) == "--benchmark";
        check(argc == 1 || timed, "usage: vectorforge_fp_state_study [--benchmark]");
        SavedEnvironment restore;
        std::cout << std::setprecision(17);
        metadata(); semantic_tests(restore.state); // performance is always gated
        if (timed) { performance(restore.state); }
        return 0;
    } catch (const std::exception& error) { std::cerr << "FP state study FAIL: " << error.what() << '\n'; return 1; }
}
