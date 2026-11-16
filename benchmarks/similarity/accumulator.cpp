#include "accumulator.hpp"

#include "accumulator_build_info.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "The accumulator study requires no-fast-math"
#endif

namespace vectorforge::benchmark::accumulator {
namespace {

using Kernel = void (*)(std::span<const float>, std::span<const float>, std::span<float>);
constexpr std::array<Kernel, 2> kernels{similarity::hardware_target::sequential_fma_scores, fourway_scores};

std::uint32_t ordered_key(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto magnitude = bits & 0x7fffffffU;
    return (bits & 0x80000000U) != 0 ? 0x80000000U - magnitude : 0x80000000U + magnitude;
}

std::uint64_t fingerprint(std::span<const float> values) {
    std::uint64_t hash = 14'695'981'039'346'656'037ULL;
    for (const auto value : values) {
        hash = (hash ^ std::bit_cast<std::uint32_t>(value)) * 1'099'511'628'211ULL;
    }
    return hash;
}

void require_bits(std::span<const float> actual, std::span<const float> expected) {
    for (std::size_t row = 0; row < expected.size(); ++row) {
        if (std::bit_cast<std::uint32_t>(actual[row]) != std::bit_cast<std::uint32_t>(expected[row])) {
            throw std::runtime_error("kernel changed its own score bits across repeated calls");
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

template <typename Range, typename Project>
std::string joined(const Range& range, Project project) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    bool first = true;
    for (const auto& value : range) {
        if (!first) { out << ';'; }
        first = false;
        out << project(value);
    }
    return out.str();
}

void write_difference(std::ostream& out, const Difference& difference) {
    out << difference.count << ',' << difference.equal_bits << ',';
    if (difference.count != 0) {
        out << static_cast<double>(difference.equal_bits) / static_cast<double>(difference.count);
    }
    out << ',' << difference.max_absolute << ',' << difference.max_relative << ',' << relative_floor << ','
        << difference.near_zero_reference << ',' << difference.max_ulp;
}

void write_environment(std::ostream& out, const Metadata& info) {
    out << quote(info.compiler) << ',' << quote(info.build_type) << ',' << quote(info.optimization_flags) << ','
        << quote(info.cpu) << ',' << quote(info.os) << ',' << quote(info.environment) << ','
        << quote(info.timestamp_utc) << ',' << (info.sanitized ? 1 : 0);
}

} // namespace

Difference differences(std::span<const float> reference, std::span<const float> experimental, double floor) {
    if (reference.size() != experimental.size() || !std::isfinite(floor) || floor <= 0) {
        throw std::invalid_argument("require equal score counts and a finite positive relative floor");
    }
    Difference result;
    result.count = reference.size();
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const float a = reference[i], b = experimental[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            throw std::invalid_argument("numerical metrics require finite scores");
        }
        result.equal_bits += std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b) ? 1U : 0U;
        const double absolute = std::abs(static_cast<double>(a) - static_cast<double>(b));
        const double magnitude = std::abs(static_cast<double>(a));
        const double relative = absolute / std::max(magnitude, floor);
        if (!std::isfinite(relative)) { throw std::overflow_error("relative difference is not representable"); }
        result.near_zero_reference += magnitude < floor ? 1U : 0U;
        result.max_absolute = std::max(result.max_absolute, absolute);
        result.max_relative = std::max(result.max_relative, relative);
        const auto x = ordered_key(a), y = ordered_key(b);
        result.max_ulp = std::max(result.max_ulp, static_cast<std::uint64_t>(x > y ? x - y : y - x));
    }
    return result;
}

Ranking compare_rankings(std::span<const RecordID> ids, std::span<const float> sequential,
                         std::span<const float> fourway, std::size_t k) {
    if (ids.size() != sequential.size()) { throw std::invalid_argument("IDs and scores must align"); }
    (void)differences(sequential, fourway); // Checks equal counts and finite scores.
    std::unordered_set<RecordID> seen;
    for (const auto id : ids) {
        if (!seen.insert(id).second) { throw std::invalid_argument("ranking IDs must be unique"); }
    }
    auto rank = [&](std::span<const float> scores) {
        std::vector<ScoredRecord> records;
        for (std::size_t i = 0; i < ids.size(); ++i) { records.push_back({ids[i], scores[i]}); }
        std::sort(records.begin(), records.end(), [](const auto& a, const auto& b) {
            return a.score > b.score || (a.score == b.score && a.id < b.id);
        });
        if (records.size() > k) { records.resize(k); }
        return records;
    };
    Ranking result{rank(sequential), rank(fourway), 0, false, false};
    seen.clear();
    for (const auto& record : result.sequential) { seen.insert(record.id); }
    for (const auto& record : result.fourway) { result.common_members += seen.contains(record.id) ? 1U : 0U; }
    result.membership_changed = result.common_members != result.sequential.size();
    result.id_order_changed = !std::equal(result.sequential.begin(), result.sequential.end(), result.fourway.begin(),
        [](const auto& a, const auto& b) { return a.id == b.id; });
    return result;
}

std::vector<RankingCase> ranking_study() {
    if (!similarity::fma::hardware_supported()) { throw std::runtime_error("AVX/FMA unavailable"); }
    const std::vector<float> query(8, 1.0F);
    const float half_ulp = std::ldexp(1.0F, -24);
    const std::array<float, 8> sensitive{1, half_ulp, 0, 0, 0, half_ulp, 0, 0};
    const std::array<float, 8> one{1, 0, 0, 0, 0, 0, 0, 0};
    auto next = one;
    next[0] = std::nextafter(1.0F, 2.0F);
    std::vector<RankingCase> cases;
    auto add = [&](std::string id, std::vector<RecordID> ids, std::size_t k, bool control) {
        std::vector<float> embeddings;
        const std::array rows{sensitive, one, next};
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const auto& row = control ? one : rows[i];
            embeddings.insert(embeddings.end(), row.begin(), row.end());
        }
        RankingCase result{std::move(id), 8, k, std::move(ids), {}, {}, {}, {}};
        result.sequential_scores.resize(result.ids.size());
        result.fourway_scores.resize(result.ids.size());
        kernels[0](query, embeddings, result.sequential_scores);
        kernels[1](query, embeddings, result.fourway_scores);
        result.difference = differences(result.sequential_scores, result.fourway_scores);
        result.ranking = compare_rankings(result.ids, result.sequential_scores, result.fourway_scores, k);
        cases.push_back(std::move(result));
    };
    add("close_top1", {20, 10}, 1, false);
    add("close_full_order", {20, 10, 30}, 3, false);
    add("exact_tie_control", {30, 10, 20}, 2, true);
    // A tiny fused cancellation residual is lost by separate partial sums.
    // This also illustrates why ULP distances near zero need absolute context.
    const float a = std::ldexp(1.0F, -23);
    const std::array cancellation_query{-1.0F, 1.0F + a};
    const std::array cancellation_data{1.0F, 1.0F - a, 0.0F, 0.0F};
    RankingCase cancellation{"fused_cancellation", 2, 1, {5, 10}, std::vector<float>(2), std::vector<float>(2), {}, {}};
    kernels[0](cancellation_query, cancellation_data, cancellation.sequential_scores);
    kernels[1](cancellation_query, cancellation_data, cancellation.fourway_scores);
    cancellation.difference = differences(cancellation.sequential_scores, cancellation.fourway_scores);
    cancellation.ranking = compare_rankings(cancellation.ids, cancellation.sequential_scores, cancellation.fourway_scores, 1);
    cases.push_back(std::move(cancellation));
    return cases;
}

void validate(const Workload& workload, const Options& options) {
    similarity::validate(workload, options);
    if (options.sampling.iterations % 2 != 0) {
        throw std::invalid_argument("measured iterations must be even and >=4 for balanced order");
    }
}

Options parse_options(std::span<const std::string_view> arguments) {
    std::vector<std::string_view> forwarded(arguments.begin(), arguments.end());
    if (std::ranges::find(arguments, "--iterations") == arguments.end()) {
        forwarded.insert(forwarded.end(), {"--iterations", "8"});
    }
    auto options = similarity::parse_options(forwarded);
    accumulator::validate(options.custom.value_or(Workload{"default"}), options);
    return options;
}

Run run_workload(const Workload& workload, const Options& options) {
    accumulator::validate(workload, options);
    if (!similarity::fma::hardware_supported()) { throw std::runtime_error("AVX/FMA unavailable"); }
    const auto data = similarity::make_data(workload, options.sampling.seed);
    std::array<std::vector<float>, 2> expected{std::vector<float>(workload.rows), std::vector<float>(workload.rows)};
    for (std::size_t i = 0; i < kernels.size(); ++i) { kernels[i](data.query, data.embeddings, expected[i]); }
    Run run{data.checksum, differences(expected[0], expected[1]), {{
        {"sequential_hardware_fma", std::string(similarity::hardware_target::kernel_isa()),
            build_info::sequential_options, fingerprint(expected[0]), {}, {}},
        {"fourway_hardware_fma", std::string(fourway_isa()),
            build_info::fourway_options, fingerprint(expected[1]), {}, {}},
    }}};
    std::vector<float> scores(workload.rows);
    for (auto& measurement : run.measurements) { measurement.samples_ms.reserve(options.sampling.iterations); }
    auto sample = [&](std::size_t variant, bool measured) {
        const auto start = std::chrono::steady_clock::now();
        kernels[variant](data.query, data.embeddings, scores);
        const auto finish = std::chrono::steady_clock::now();
        require_bits(scores, expected[variant]);
        if (fingerprint(scores) != run.measurements[variant].checksum) { throw std::runtime_error("score checksum mismatch"); }
        if (measured) {
            run.measurements[variant].samples_ms.push_back(
                std::chrono::duration<double, std::milli>(finish - start).count());
        }
    };
    for (std::size_t i = 0; i < options.sampling.warmup; ++i) { sample(i % 2, false); sample(1 - i % 2, false); }
    for (std::size_t i = 0; i < options.sampling.iterations; ++i) { sample(i % 2, true); sample(1 - i % 2, true); }
    for (auto& measurement : run.measurements) { measurement.timing = benchmark::summarize(measurement.samples_ms); }
    return run;
}

void write_csv_header(std::ostream& output) {
    output << "workload,variant,n,d,embedding_bytes,warmup,iterations,seed,median_ms,min_ms,max_ms,"
              "ns_per_dot,dots_per_second,dataset_checksum,result_checksum,samples_ms,timing_scope,"
              "comparison_order,kernel_isa,kernel_compile_options,compiler,build_type,optimization_flags,"
              "cpu,os,environment,timestamp_utc,sanitized,mode,compared_scores,bitwise_equal,bitwise_equality_rate,"
              "max_absolute_difference,max_relative_difference,relative_floor,near_zero_reference,max_ulp_distance\n";
}

void write_csv(std::ostream& output, const Workload& workload, const Options& options,
               const Metadata& info, const Run& run) {
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const auto& m : run.measurements) {
        output << quote(workload.id) << ',' << quote(m.variant) << ',' << workload.rows << ',' << workload.dimension << ','
               << workload.rows * workload.dimension * sizeof(float) << ',' << options.sampling.warmup << ','
               << options.sampling.iterations << ',' << options.sampling.seed << ',' << m.timing.median_ms << ','
               << m.timing.min_ms << ',' << m.timing.max_ms << ',';
        if (workload.rows != 0) { output << m.timing.median_ms * 1e6 / static_cast<double>(workload.rows); }
        output << ',' << static_cast<double>(workload.rows) * 1000 / m.timing.median_ms << ',' << run.dataset_checksum << ','
               << m.checksum << ',' << quote(joined(m.samples_ms, [](double v) { return v; })) << ','
               << "\"one_scan_shape_checks_finite_checks_output_stores\",\"balanced_alternating_sequential_first_round0\","
               << quote(m.isa) << ',' << quote(m.compile_options) << ',';
        write_environment(output, info);
        output << ',' << quote(options.sampling.smoke ? "smoke" : "performance") << ',';
        write_difference(output, run.difference);
        output << '\n';
    }
}

void write_ranking_csv(std::ostream& output, const Metadata& info, std::span<const RankingCase> cases) {
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    output << "case,n,d,k,sequential_topk_ids,fourway_topk_ids,common_members,membership_changed,id_order_changed,"
              "candidate_ids,sequential_scores,fourway_scores,sequential_score_bits,fourway_score_bits,"
              "compared_scores,bitwise_equal,bitwise_equality_rate,max_absolute_difference,max_relative_difference,"
              "relative_floor,near_zero_reference,max_ulp_distance,kernel_isa,compiler,build_type,optimization_flags,"
              "cpu,os,environment,timestamp_utc,sanitized,mode\n";
    for (const auto& c : cases) {
        output << quote(c.id) << ',' << c.ids.size() << ',' << c.dimension << ',' << c.k << ','
               << quote(joined(c.ranking.sequential, [](const auto& r) { return r.id; })) << ','
               << quote(joined(c.ranking.fourway, [](const auto& r) { return r.id; })) << ','
               << c.ranking.common_members << ',' << c.ranking.membership_changed << ',' << c.ranking.id_order_changed << ','
               << quote(joined(c.ids, [](auto id) { return id; })) << ','
               << quote(joined(c.sequential_scores, [](float v) { return v; })) << ','
               << quote(joined(c.fourway_scores, [](float v) { return v; })) << ','
               << quote(joined(c.sequential_scores, [](float v) { return std::bit_cast<std::uint32_t>(v); })) << ','
               << quote(joined(c.fourway_scores, [](float v) { return std::bit_cast<std::uint32_t>(v); })) << ',';
        write_difference(output, c.difference);
        output << ',' << quote(fourway_isa()) << ',';
        write_environment(output, info);
        output << ",numerical_analysis\n";
    }
}

} // namespace vectorforge::benchmark::accumulator
