#include "fma_comparison.hpp"

#include "fma_build_info.hpp"

#include <bit>
#include <chrono>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "The FMA comparison requires no-fast-math"
#endif

namespace vectorforge::benchmark::similarity::fma {
namespace {

using Kernel = void (*)(std::span<const float>, std::span<const float>, std::span<float>);
constexpr std::array<Kernel, 2> kernels{generic_target::sequential_fma_scores,
                                       hardware_target::sequential_fma_scores};

std::uint64_t fingerprint(std::span<const float> values) {
    std::uint64_t hash = 14'695'981'039'346'656'037ULL;
    for (const auto value : values) {
        hash = (hash ^ std::bit_cast<std::uint32_t>(value)) * 1'099'511'628'211ULL;
    }
    return hash;
}

void exact_scores(std::span<const float> actual, std::span<const float> expected) {
    for (std::size_t row = 0; row < expected.size(); ++row) {
        if (std::bit_cast<std::uint32_t>(actual[row]) != std::bit_cast<std::uint32_t>(expected[row])) {
            throw std::runtime_error("target comparison float32 score-bit mismatch");
        }
    }
}

std::string quote(std::string_view value) {
    std::string result = "\"";
    for (const auto c : value) {
        if (c == '"') { result += '"'; }
        result += c;
    }
    return result + '"';
}

} // namespace

bool hardware_supported() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx") && __builtin_cpu_supports("fma");
}

Run run_workload(const Workload& workload, const Options& options) {
    validate(workload, options);
    if (!hardware_supported()) { throw std::runtime_error("AVX/FMA unavailable; hardware kernel was not called"); }
    const auto data = make_data(workload, options.sampling.seed);
    std::vector<float> expected(workload.rows);
    // The unchanged V1-01 path is itself checked against production by its tests.
    // Target-specific tests additionally compare both targets to production.
    similarity::sequential_fma_scores(data.query, data.embeddings, expected);
    std::vector<float> scores(workload.rows);
    Run run{data.checksum, fingerprint(expected), {{
        {"generic_x86_64", std::string(generic_target::kernel_isa()), build_info::generic_options, {}, {}},
        {"hardware_fma", std::string(hardware_target::kernel_isa()), build_info::hardware_options, {}, {}},
    }}};
    for (auto& measurement : run.measurements) { measurement.samples_ms.reserve(options.sampling.iterations); }
    // Correctness before warm-up, and after every later call. No mismatches are
    // tolerated or hidden by checksums, averaging or ranking tolerances.
    for (const auto kernel : kernels) {
        kernel(data.query, data.embeddings, scores);
        exact_scores(scores, expected);
    }
    auto consume = [&](std::size_t target, bool measured) {
        const auto start = std::chrono::steady_clock::now();
        kernels[target](data.query, data.embeddings, scores);
        const auto finish = std::chrono::steady_clock::now();
        exact_scores(scores, expected);
        if (fingerprint(scores) != run.result_checksum) { throw std::runtime_error("target result checksum mismatch"); }
        if (measured) {
            run.measurements[target].samples_ms.push_back(
                std::chrono::duration<double, std::milli>(finish - start).count());
        }
    };
    for (std::size_t i = 0; i < options.sampling.warmup; ++i) {
        consume(i % 2, false);
        consume(1 - i % 2, false);
    }
    for (std::size_t i = 0; i < options.sampling.iterations; ++i) {
        consume(i % 2, true);
        consume(1 - i % 2, true);
    }
    for (auto& measurement : run.measurements) { measurement.timing = benchmark::summarize(measurement.samples_ms); }
    return run;
}

void write_csv_header(std::ostream& output) {
    output << "workload,variant,n,d,embedding_bytes,warmup,iterations,seed,median_ms,min_ms,max_ms,"
              "ns_per_dot,dots_per_second,dataset_checksum,result_checksum,samples_ms,timing_scope,"
              "comparison_order,kernel_isa,kernel_compile_options,compiler,build_type,optimization_flags,"
              "cpu,os,environment,timestamp_utc,sanitized,mode\n";
}

void write_csv(std::ostream& output, const Workload& workload, const Options& options,
               const Metadata& info, const Run& run) {
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const auto& measurement : run.measurements) {
        std::ostringstream samples;
        samples.imbue(std::locale::classic());
        samples << std::setprecision(std::numeric_limits<double>::max_digits10);
        for (std::size_t i = 0; i < measurement.samples_ms.size(); ++i) {
            if (i != 0) { samples << ';'; }
            samples << measurement.samples_ms[i];
        }
        output << quote(workload.id) << ',' << quote(measurement.variant) << ',' << workload.rows << ','
               << workload.dimension << ',' << workload.rows * workload.dimension * sizeof(float) << ','
               << options.sampling.warmup << ',' << options.sampling.iterations << ',' << options.sampling.seed << ','
               << measurement.timing.median_ms << ',' << measurement.timing.min_ms << ',' << measurement.timing.max_ms << ',';
        if (workload.rows != 0) { output << measurement.timing.median_ms * 1e6 / static_cast<double>(workload.rows); }
        output << ',' << static_cast<double>(workload.rows) * 1000 / measurement.timing.median_ms << ','
               << run.dataset_checksum << ',' << run.result_checksum << ',' << quote(samples.str()) << ','
               << "\"one_row_major_scan_shape_checks_finite_checks_output_stores\",\"alternate_first_each_round_generic_first_round0\","
               << quote(measurement.isa) << ',' << quote(measurement.compile_options) << ','
               << quote(info.compiler) << ',' << quote(info.build_type) << ',' << quote(info.optimization_flags) << ','
               << quote(info.cpu) << ',' << quote(info.os) << ',' << quote(info.environment) << ','
               << quote(info.timestamp_utc) << ',' << (info.sanitized ? 1 : 0) << ','
               << quote(options.sampling.smoke ? "smoke" : "performance") << '\n';
    }
}

} // namespace vectorforge::benchmark::similarity::fma
