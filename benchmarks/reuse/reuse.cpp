#include "reuse.hpp"

#include "vectorforge/batched_executor.hpp"
#include "vectorforge/validated_dataset.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace vectorforge::benchmark::reuse {
namespace {

std::uint64_t fingerprint(std::span<const ScoredRecord> records) {
    std::uint64_t hash = 14'695'981'039'346'656'037ULL;
    for (const auto& record : records) {
        hash = (hash ^ record.id) * 1'099'511'628'211ULL;
        hash = (hash ^ std::bit_cast<std::uint32_t>(record.score)) * 1'099'511'628'211ULL;
    }
    return hash;
}

void exact_results(std::span<const ScoredRecord> actual, std::span<const ScoredRecord> expected) {
    if (actual.size() != expected.size()) { throw std::runtime_error("trusted/legacy result count mismatch"); }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i].id != expected[i].id ||
            std::bit_cast<std::uint32_t>(actual[i].score) != std::bit_cast<std::uint32_t>(expected[i].score)) {
            throw std::runtime_error("trusted/legacy ID, ordering or float32 score-bit mismatch");
        }
    }
}

std::string csv_quote(std::string_view value) {
    std::string result = "\"";
    for (const auto c : value) {
        if (c == '"') { result += '"'; }
        result += c;
    }
    return result + '"';
}

std::string sample_text(const Samples& samples) {
    std::ostringstream text;
    text.imbue(std::locale::classic());
    text << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (std::size_t i = 0; i < samples.values_ms.size(); ++i) {
        if (i != 0) { text << ';'; }
        text << samples.values_ms[i];
    }
    return text.str();
}

} // namespace

void validate(const Workload& workload, const Options& options) {
    benchmark::validate(workload.query, options.sampling);
    if (workload.reuse_counts.empty() || std::ranges::find(workload.reuse_counts, 0) != workload.reuse_counts.end()) {
        throw std::invalid_argument("reuse counts must be positive and nonempty");
    }
}

Options parse_options(std::span<const std::string_view> arguments) {
    Options options;
    std::vector<std::string_view> forwarded;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        if (arguments[i] != "--reuse") { forwarded.push_back(arguments[i]); continue; }
        if (++i == arguments.size()) { throw std::invalid_argument("missing --reuse value"); }
        const auto text = arguments[i];
        std::size_t value = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (error != std::errc{} || end != text.data() + text.size() || value == 0) {
            throw std::invalid_argument("reuse must be a positive decimal size_t integer");
        }
        options.custom_reuse = value;
    }
    options.sampling = benchmark::parse_options(forwarded);
    if (options.sampling.smoke && options.custom_reuse) {
        throw std::invalid_argument("--smoke uses a fixed tiny reuse case");
    }
    validate({options.sampling.custom.value_or(benchmark::Workload{"default"}),
              {options.custom_reuse.value_or(10)}}, options);
    return options;
}

std::vector<Workload> default_workloads() {
    return {
        {{"small_low", 10'000, 32, 10, 10, 256}, {1, 10, 100}},
        {{"larger_low", 50'000, 32, 10, 10, 256}, {1, 10, 100}},
        {{"higher_dimension_low", 10'000, 128, 10, 10, 256}, {1, 10, 100}},
        {{"small_high", 10'000, 32, 100, 10, 256}, {1, 10, 100}},
    };
}

Model model(double admission_ms, double legacy_ms, double trusted_ms, std::size_t reuse_count) {
    if (reuse_count == 0) { throw std::invalid_argument("modeled reuse count must be positive"); }
    for (const auto value : std::array{admission_ms, legacy_ms, trusted_ms}) {
        if (!std::isfinite(value) || value < 0) {
            throw std::invalid_argument("model latencies must be finite and nonnegative");
        }
    }
    const double legacy_total = static_cast<double>(reuse_count) * legacy_ms;
    const double trusted_total = admission_ms + static_cast<double>(reuse_count) * trusted_ms;
    if (!std::isfinite(legacy_total) || !std::isfinite(trusted_total)) {
        throw std::overflow_error("modeled workflow total is not representable");
    }
    std::optional<std::uint64_t> threshold;
    if (trusted_ms < legacy_ms) {
        const long double savings = static_cast<long double>(legacy_ms) - static_cast<long double>(trusted_ms);
        const long double admission = admission_ms;
        const long double first = std::floor(admission / savings) + 1;
        if (!std::isfinite(first) || first >= std::ldexp(1.0L, 64)) {
            throw std::overflow_error("break-even reuse count exceeds uint64_t");
        }
        auto count = std::max<std::uint64_t>(1, static_cast<std::uint64_t>(first));
        // Enforce strict inequality and minimality at integral boundaries using
        // long-double arithmetic on the measured binary64 medians.
        while (count > 1 && static_cast<long double>(count - 1) * savings > admission) { --count; }
        while (!(static_cast<long double>(count) * savings > admission)) {
            if (count == std::numeric_limits<std::uint64_t>::max()) {
                throw std::overflow_error("break-even reuse count exceeds uint64_t");
            }
            ++count;
        }
        threshold = count;
    }
    return {legacy_total, trusted_total, threshold};
}

Run run_workload(const Workload& workload, const Options& options) {
    validate(workload, options);
    const auto& query_config = workload.query;
    const auto& sampling = options.sampling;
    const auto source = make_dataset(query_config, sampling.seed);
    const std::array predicates{Predicate{0, Comparison::equal, std::int32_t{1}}};
    auto legacy_query = [&] {
        return execute_batched(source.columns, source.embeddings, source.ids, source.query,
                               predicates, query_config.k, query_config.batch_size);
    };
    const auto expected = legacy_query(); // Correctness oracle outside all timing.
    const auto checksum = fingerprint(expected);
    Run run{source.qualifying, expected.size(), source.checksum, checksum, {}, {}, {}};
    for (auto* samples : {&run.admission, &run.legacy_query, &run.trusted_query}) {
        samples->values_ms.reserve(sampling.iterations);
    }
    std::optional<ValidatedDataset> admitted;
    auto admission_sample = [&](bool measured) {
        admitted.reset(); // Retire the previous snapshot before starting the clock.
        const auto start = std::chrono::steady_clock::now();
        admitted.emplace(source.columns, source.embeddings, source.ids);
        const auto finish = std::chrono::steady_clock::now();
        exact_results(execute_batched(*admitted, source.query, predicates, query_config.k, query_config.batch_size), expected);
        if (!source.ids.empty() && admitted->record_ids().data() == source.ids.data()) {
            throw std::runtime_error("admission unexpectedly reused mutable source ID storage");
        }
        if (measured) {
            run.admission.values_ms.push_back(std::chrono::duration<double, std::milli>(finish - start).count());
        }
    };
    for (std::size_t i = 0; i < sampling.warmup; ++i) { admission_sample(false); }
    for (std::size_t i = 0; i < sampling.iterations; ++i) { admission_sample(true); }
    // Keep the last measured admission. No subsequent query constructs a snapshot.
    const auto& trusted = *admitted;
    const auto* trusted_ids = trusted.record_ids().data();
    auto query_sample = [&](std::size_t path, bool measured) {
        const auto start = std::chrono::steady_clock::now();
        const auto result = path == 0 ? legacy_query()
            : execute_batched(trusted, source.query, predicates, query_config.k, query_config.batch_size);
        const auto finish = std::chrono::steady_clock::now();
        exact_results(result, expected);
        if (fingerprint(result) != checksum || trusted.record_ids().data() != trusted_ids) {
            throw std::runtime_error("query result/storage changed during reuse");
        }
        if (measured) {
            auto& samples = path == 0 ? run.legacy_query : run.trusted_query;
            samples.values_ms.push_back(std::chrono::duration<double, std::milli>(finish - start).count());
        }
    };
    for (std::size_t i = 0; i < sampling.warmup; ++i) {
        query_sample(i % 2, false);
        query_sample(1 - i % 2, false);
    }
    for (std::size_t i = 0; i < sampling.iterations; ++i) {
        query_sample(i % 2, true);
        query_sample(1 - i % 2, true);
    }
    for (auto* samples : {&run.admission, &run.legacy_query, &run.trusted_query}) {
        samples->timing = summarize(samples->values_ms);
    }
    return run;
}

void write_csv_header(std::ostream& output) {
    output << "workload,n,d,selectivity_percent,actual_selectivity,k,batch_size,reuse_count,qualifying_candidates,"
              "returned_results,warmup,iterations,seed,admission_median_ms,admission_min_ms,admission_max_ms,"
              "legacy_query_median_ms,legacy_query_min_ms,legacy_query_max_ms,trusted_query_median_ms,"
              "trusted_query_min_ms,trusted_query_max_ms,legacy_total_ms_model,trusted_total_ms_model,total_kind,"
              "break_even_reuse_model,break_even_status,dataset_checksum,result_checksum,admission_samples_ms,"
              "legacy_query_samples_ms,trusted_query_samples_ms,compiler,build_type,optimization_flags,cpu,os,"
              "environment,timestamp_utc,sanitized,mode\n";
}

void write_csv(std::ostream& output, const Workload& workload, const Options& options,
               const Metadata& info, const Run& run) {
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    const auto& query = workload.query;
    const auto& sampling = options.sampling;
    const auto& admission = run.admission.timing;
    const auto& legacy = run.legacy_query.timing;
    const auto& trusted = run.trusted_query.timing;
    const double selectivity = query.rows == 0 ? 0 : static_cast<double>(run.qualifying) / static_cast<double>(query.rows);
    for (const auto count : workload.reuse_counts) {
        const auto totals = model(admission.median_ms, legacy.median_ms, trusted.median_ms, count);
        output << csv_quote(query.id) << ',' << query.rows << ',' << query.dimension << ',' << query.selectivity_percent << ','
               << selectivity << ',' << query.k << ',' << query.batch_size << ',' << count << ',' << run.qualifying << ','
               << run.returned << ',' << sampling.warmup << ',' << sampling.iterations << ',' << sampling.seed << ','
               << admission.median_ms << ',' << admission.min_ms << ',' << admission.max_ms << ','
               << legacy.median_ms << ',' << legacy.min_ms << ',' << legacy.max_ms << ','
               << trusted.median_ms << ',' << trusted.min_ms << ',' << trusted.max_ms << ','
               << totals.legacy_total_ms << ',' << totals.trusted_total_ms << ",model_from_component_medians,";
        if (totals.break_even_reuse) { output << *totals.break_even_reuse; }
        output << ',' << (totals.break_even_reuse ? "finite" : "trusted_not_faster") << ','
               << run.dataset_checksum << ',' << run.result_checksum << ',' << csv_quote(sample_text(run.admission)) << ','
               << csv_quote(sample_text(run.legacy_query)) << ',' << csv_quote(sample_text(run.trusted_query)) << ','
               << csv_quote(info.compiler) << ',' << csv_quote(info.build_type) << ',' << csv_quote(info.optimization_flags) << ','
               << csv_quote(info.cpu) << ',' << csv_quote(info.os) << ',' << csv_quote(info.environment) << ','
               << csv_quote(info.timestamp_utc) << ',' << (info.sanitized ? 1 : 0) << ','
               << csv_quote(sampling.smoke ? "smoke" : "performance") << '\n';
    }
}

} // namespace vectorforge::benchmark::reuse
