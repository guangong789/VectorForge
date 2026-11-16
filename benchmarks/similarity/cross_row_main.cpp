#include "cross_row.hpp"
#include "benchmark.hpp"
#include "cross_row_build_info.hpp"

#include <array>
#include <bit>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>

namespace {
namespace bench = vectorforge::benchmark;
namespace study = bench::cross_row;
constexpr std::size_t windows = 16, repeats = 16, warmup = 2, iterations = 8;
constexpr std::uint32_t seed = 0x56463107U;
using Schedule = std::array<std::array<std::size_t, study::capacity>, windows>;
using Scores = std::array<std::array<float, study::capacity>, windows>;
constexpr std::array<study::Kernel, 2> kernels{study::scalar_scores, study::avx2_scores};

std::string quoted(std::string_view value) {
    std::string result = "\"";
    for (const auto c : value) { if (c == '"') { result += '"'; } result += c; }
    return result + '"';
}
std::uint64_t hash(std::span<const float> values, std::uint64_t state = 14695981039346656037ULL) {
    for (const auto v : values) { state = (state ^ std::bit_cast<std::uint32_t>(v)) * 1099511628211ULL; }
    return state;
}
void exact(const Scores& a, const Scores& b) {
    for (std::size_t w = 0; w < windows; ++w) {
        for (std::size_t i = 0; i < study::capacity; ++i) {
            if (std::bit_cast<std::uint32_t>(a[w][i]) != std::bit_cast<std::uint32_t>(b[w][i])) {
                throw std::runtime_error("cross-row benchmark exact score mismatch");
            }
        }
    }
}

void workload(std::size_t rows, std::size_t dimension, std::size_t group, bool sparse,
              const bench::Metadata& meta, bool smoke) {
    std::uint32_t state = seed;
    auto random_value = [&] {
        state = state * 1664525U + 1013904223U;
        return static_cast<float>(static_cast<int>((state >> 8) % 20001) - 10000) / 8191.0F;
    };
    std::vector<float> values(rows * dimension), query(dimension);
    std::generate(values.begin(), values.end(), random_value);
    std::generate(query.begin(), query.end(), random_value);
    auto data_hash = hash(query, hash(values));
    const vectorforge::EmbeddingBuffer buffer(rows, dimension, std::move(values));
    Schedule schedule{};
    for (std::size_t w = 0; w < windows; ++w) {
        for (std::size_t i = 0; i < group; ++i) {
            const auto ordinal = w * group + i;
            schedule[w][i] = sparse ? (ordinal * 263 + 17) % rows : ordinal % rows;
            data_hash = (data_hash ^ schedule[w][i]) * 1099511628211ULL;
        }
    }
    Scores expected{}, output{};
    for (std::size_t w = 0; w < windows; ++w) {
        if (study::scalar_scores(buffer, query, std::span(schedule[w]).first(group),
                                 std::span(expected[w]).first(group)) != group) {
            throw std::runtime_error("unexpected generated-input overflow");
        }
    }
    std::array<std::vector<double>, 2> samples;
    for (auto& s : samples) { s.reserve(iterations); }
    const auto batch_repeats = smoke ? 1U : repeats;
    auto sample = [&](std::size_t variant, bool measured) {
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t r = 0; r < batch_repeats; ++r) {
            for (std::size_t w = 0; w < windows; ++w) {
                if (kernels[variant](buffer, query, std::span(schedule[w]).first(group),
                                     std::span(output[w]).first(group)) != group) {
                    throw std::runtime_error("unexpected timed scoring overflow");
                }
            }
        }
        const auto stop = std::chrono::steady_clock::now();
        exact(output, expected); // All stores observable; outside timing.
        if (measured) { samples[variant].push_back(std::chrono::duration<double, std::milli>(stop - start).count()); }
    };
    for (std::size_t r = 0; r < warmup; ++r) { sample(r % 2, false); sample(1 - r % 2, false); }
    for (std::size_t r = 0; r < iterations; ++r) { sample(r % 2, true); sample(1 - r % 2, true); }
    std::uint64_t result_hash = 14695981039346656037ULL;
    for (const auto& scores : expected) { result_hash = hash(scores, result_hash); }
    const auto scored = static_cast<double>(windows * batch_repeats * group);
    for (std::size_t v = 0; v < 2; ++v) {
        const auto timing = bench::summarize(samples[v]);
        std::ostringstream raw;
        raw << std::setprecision(17);
        for (std::size_t i = 0; i < samples[v].size(); ++i) { if (i) { raw << ';'; } raw << samples[v][i]; }
        std::cout << rows << ',' << dimension << ',' << group << ',' << (sparse ? "sparse" : "contiguous") << ','
            << (v ? "avx2_gather" : "scalar_fma") << ',' << rows * dimension * sizeof(float) << ','
            << std::min(rows, windows * group) << ',' << windows << ',' << batch_repeats << ',' << warmup << ','
            << iterations << ',' << seed << ',' << timing.median_ms << ',' << timing.min_ms << ',' << timing.max_ms << ','
            << timing.median_ms * 1e6 / scored << ',' << scored * 1000 / timing.median_ms << ','
            << data_hash << ',' << result_hash << ',' << quoted(raw.str()) << ','
            << "\"shape_and_index_checks_gather_or_load_fma_per_step_finite_stores\",\"balanced_alternating_scalar_first_round0\","
            << quoted(v ? study::avx2_options : study::scalar_options) << ',' << quoted(meta.compiler) << ','
            << quoted(meta.build_type) << ',' << quoted(meta.optimization_flags) << ',' << quoted(meta.cpu) << ','
            << quoted(meta.os) << ',' << quoted(meta.environment) << ',' << quoted(meta.timestamp_utc) << ','
            << std::fegetround() << ',' << study::environment() << ',' << (meta.sanitized ? 1 : 0) << ','
            << (smoke ? "smoke" : "performance") << '\n';
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const bool smoke = argc == 2 && std::string_view(argv[1]) == "--smoke";
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            std::cout << "Usage: vectorforge_cross_row_benchmark [--smoke]\n"
                "Default: N=256/8192, D=32/128/768, groups=7/32, contiguous/sparse (24 cases).\n"
                "16 fixed windows; 16 repeats/sample, 2 warm-ups, 8 balanced samples per variant.\n"
                "Requires AVX2/FMA, nearest rounding, gradual underflow, masked exceptions.\n";
            return 0;
        }
        if (argc != 1 && !smoke) { throw std::invalid_argument("use --help or --smoke"); }
        if (!study::supported()) { std::cerr << "SKIP: AVX2/FMA unavailable\n"; return 77; }
        if (!study::supported_environment()) {
            std::cerr << "SKIP: require nearest, FTZ/DAZ off, masked exceptions\n"; return 77;
        }
        const auto meta = bench::metadata();
        bench::require_performance_build(meta, smoke);
        std::cout.imbue(std::locale::classic());
        std::cout << std::setprecision(17)
            << "n,d,group,selection,variant,embedding_bytes,unique_selected_rows,windows,repeats,warmup,iterations,seed,"
               "median_ms,min_ms,max_ms,ns_per_score,scores_per_second,data_checksum,result_checksum,samples_ms,"
               "timing_scope,comparison_order,kernel_compile_options,compiler,build_type,global_and_core_flags,"
               "cpu,os,environment,timestamp_utc,rounding_mode,mxcsr,sanitized,mode\n";
        if (smoke) {
            workload(37, 9, 9, true, meta, true);
        } else {
            for (const auto rows : {256U, 8192U}) {
                for (const auto d : {32U, 128U, 768U}) {
                    for (const auto group : {7U, 32U}) {
                        for (const bool sparse : {false, true}) { workload(rows, d, group, sparse, meta, false); }
                    }
                }
            }
        }
        if (!std::cout) { throw std::runtime_error("cannot write cross-row CSV"); }
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
